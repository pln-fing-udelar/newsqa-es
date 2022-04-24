#!/usr/bin/env python3

from transformers import MarianTokenizer, MarianMTModel
from os import listdir
from os.path import isfile, join

import torch
import torch.multiprocessing as mp

def chunks(lst, n):
    # Yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

path_stories = "cnn_stories/cnn/stories/"
path_translated = "cnn_stories/cnn/translated/"
model_name = "Helsinki-NLP/opus-mt-en-es"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Download the model and the tokenizer
model = MarianMTModel.from_pretrained(model_name).to(device)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_story(file_name):
	try:
		file_story = open(path_stories + file_name, 'r')
		file_translated = open(path_translated + file_name, "w")
		text = file_story.readlines()
		lines = []
		for line in text:
			if line != '\n':
				lines.append(line)
		for ck in chunks(lines, 20):
			# Tokenize the text
			batch = tokenizer(ck, return_tensors="pt", padding=True).to(device)
			batch["input_ids"] = batch["input_ids"][:, :512]
			batch["attention_mask"] = batch["attention_mask"][:, :512]
			# Translate
			translation = model.generate(**batch)
			out_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
			for out_line in out_text:
				file_translated.write(out_line + '\n')
		file_translated.close()
		file_story.close()
	except RuntimeError as e:
		print("Unable to translate", file_name, e)
		raise

if __name__ == "__main__":
	onlyfiles = [f for f in listdir(path_stories) if isfile(join(path_stories, f))]
	if device == 'cuda':
		# combining with multiprocessing works but gives OOM in my GPU
		for f in onlyfiles:
			translate_story(f)
	else:
		mp.set_start_method('spawn')
		with mp.Pool(processes=max(1, mp.cpu_count()-1)) as pool:
			pool.map(translate_story, onlyfiles)
