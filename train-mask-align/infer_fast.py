#!/usr/bin/env python
# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import socket
import string
import shutil
import argparse
import subprocess
import numpy as np
import thualign.data as data
import thualign.models as models
import thualign.utils as utils
import thualign.utils.alignment as alignment_utils

from nltk.translate import Alignment

import torch
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate alignments neural alignment models",
        usage="inferrer.py [<args>] [-h | --help]"
    )
    # test args
    parser.add_argument("--alignment-output", type=str, help="path to save generated alignments")

    # configure file
    parser.add_argument("--config", type=str, required=True,
                        help="Provided config file")
    parser.add_argument("--base-config", type=str, help="base config file")
    parser.add_argument("--data-config", type=str, help="data config file")
    parser.add_argument("--model-config", type=str, help="base config file")
    parser.add_argument("--exp", "-e", default='DEFAULT', type=str, help="name of experiments")

    return parser.parse_args()

def load_vocabulary(params):
    params.vocabulary = {
        "source": data.Vocabulary(params.vocab[0]), 
        "target": data.Vocabulary(params.vocab[1])
    }
    return params

def to_cuda(features):
    for key in features:
        features[key] = features[key].cuda()

    return features

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

def get_first_greater_than(l, threshold):
    for idx, val in enumerate(l):
        if val > threshold:
            return idx
    return -1

def get_last_greater_than(l, threshold):
    last_idx = -1
    for idx, val in enumerate(l):
        if val > threshold:
            last_idx = idx
    return last_idx
    
def get_answer_token_indexes(ans_idxs, tokens):
    idx1 = -1
    idx2 = -1
    current_char = 0
    search1 = True

    char_idx1, char_idx2 = int(ans_idxs[0].split(":")[0]), int(ans_idxs[0].split(":")[1])
    
    for idx, tok in enumerate(tokens):
        current_char += len(tok) + 1
        if char_idx1 < current_char and search1:
            idx1 = idx
            search1 = False
        if char_idx2 < current_char:
            idx2 = idx
            break

    return idx1, idx2 + 1
    

def gen_align(params):
    """Generate alignments
    """
        
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
    dist.init_process_group("nccl", init_method=url,
                            rank=0,
                            world_size=1)

    params = load_vocabulary(params)
    checkpoint = getattr(params, "checkpoint", None) or utils.best_checkpoint(params.output)

    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if params.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    # Create model
    with torch.no_grad():

        model = models.get_model(params).cuda()

        if params.half:
            model = model.half()
        
        model.eval()
        print('loading checkpoint: {}'.format(checkpoint))
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

        get_infer_dataset = data.AlignmentPipeline.get_infer_dataset
        dataset = get_infer_dataset(params.test_input, params)

        dataset = torch.utils.data.DataLoader(dataset, batch_size=None)
        iterator = iter(dataset)
        counter = 0

        # Buffers for synchronization
        results = [0., 0.]
        
        extract_params = alignment_utils.get_extract_params(params)

        print(f"src_file: {os.path.abspath(params.test_input[0])}\n"
              f"tgt_file: {os.path.abspath(params.test_input[1])}\n"
              f"ans_file: {os.path.abspath(params.test_answers)}\n"
              f"alignment_output: {os.path.abspath(params.alignment_output)}")

        src_file = open(params.test_input[0], encoding="utf8")
        tgt_file = open(params.test_input[1], encoding="utf8")
        
        ans_file = open(params.test_answers, encoding="utf8")

        output_file = open(params.alignment_output, 'w', encoding="utf8")

        while True:
            try:
                # get one batch of data
                features = next(iterator)
                features = to_cuda(features)
            except:
                break

            t = time.time()
            counter += 1

            # run mask-predict
            acc_cnt, all_cnt, state = model.cal_alignment(features)
            score = 0.0 if all_cnt == 0 else acc_cnt / all_cnt

            results[0] += acc_cnt
            results[1] += all_cnt
            
            source_lengths, target_lengths = features["source_mask"].sum(-1).long().tolist(), features["target_mask"].sum(-1).long().tolist()
            
            for weight_f, weight_b, src_len, tgt_len in zip(state['f_cross_attn'], state['b_cross_attn'], source_lengths, target_lengths):
                src = src_file.readline().strip().split()
                tgt = tgt_file.readline().strip().split()
                ans = ans_file.readline().strip().split()
                
                # The "ans" file contains the position of the answer in the format idx1:idx2
                answer_position = get_answer_token_indexes(ans, tgt)
                  
                # calculate alignment scores (weight_final) for each sentence pair
                weight_f, weight_b = weight_f.detach(), weight_b.detach()
                weight_f, weight_b = weight_f[-1].mean(dim=0)[:tgt_len, :src_len], weight_b[-1].mean(dim=0)[:tgt_len, :src_len]
                weight_final = 2*(weight_f * weight_b)/(weight_f + weight_b)
                
                # keep only relevant rows (the ones corresponding to the answer) and normalize
                weight_added_per_word = weight_final[answer_position[0]:answer_position[1]].sum(dim=0) / weight_final.sum(dim=0)

                threshold = 0.3932

                first_word_in_answer = get_first_greater_than(weight_added_per_word, threshold)
                last_word_in_answer = get_last_greater_than(weight_added_per_word, threshold)

                result = ""
                if first_word_in_answer != -1 and last_word_in_answer != -1:
                    min_idx = max(0, first_word_in_answer)
                    max_idx = min(len(src), last_word_in_answer+1)

                    #Brackets in sentence
                    src.insert(max_idx, "}}")
                    src.insert(min_idx, "{{")
                    result = " ".join(src)
                    
                output_file.write(result + '\n')
                
            t = time.time() - t
            print("Finished batch(%d): %.3f (%.3f sec)" % (counter, score, t))
                
        score = 0.0 if results[1] == 0 else results[0] / results[1]
        print("acc_rate: %f" % (score))
        
def main(args):
    params = utils.Config.read(args.config, base=args.base_config, data=args.data_config, model=args.model_config, exp=args.exp)
    params.alignment_output = args.alignment_output 
    gen_align(params)
        
if __name__ == "__main__":
    main(parse_args())
