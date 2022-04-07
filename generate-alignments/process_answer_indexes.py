import os
import re
import random

### This script is used to generate the following files (using the files created by the preprocess_anotated.py script): 
###    - answers.en:   contains the index of the answer for each sentence, after tokenizing with sentencepiece
###    - answers.32k.en.txt:   contains the answer for each sentence in plain text, after tokenizing with sentencepiece
###    - answers.en.txt:   contains the answer for each sentence in plain text, untokenized


def process_answer_indexes(sent_untok_filename, sent_tok_filename, ans_idx_untok_filename, ans_idx_tok_filename, ans_tok_filename, ans_untok_filename):

    sentences_untokenized = open("./" + sent_untok_filename, "r", encoding="utf-8")
    sentences_tokenized = open("./" + sent_tok_filename, "r", encoding="utf-8")
    answer_indexes_untokenized = open("./" + ans_idx_untok_filename, "r", encoding="utf-8")
    answer_indexes_tokenized = open("./" + ans_idx_tok_filename, "w", encoding="utf-8")
    answer_tokenized = open("./" + ans_tok_filename, "w", encoding="utf-8")
    answer_untokenized = open("./" + ans_untok_filename, "w", encoding="utf-8")

    data1 = sentences_untokenized.readlines()
    data2 = sentences_tokenized.readlines()
    data3 = answer_indexes_untokenized.readlines()
    num_sentences = min(min(len(data1), len(data2)), len(data3))

    for i in range(0, num_sentences):
        sentence_untok = data1[i]
        sentence_tok = data2[i]
        
        sentence_untok = sentence_untok.replace("$", r"Ç")
        sentence_tok = sentence_tok.replace("$", r"Ç")
        
        idx1 = int(data3[i].split(':')[0])
        idx2 = int(data3[i].split(':')[1])
        
        ans = r'{}'.format(sentence_untok[idx1:idx2])
        ans_fixed = re.sub("\?", "\?", ans)
        ans_fixed = re.sub("\.", "\.", ans_fixed)
        ans_fixed = re.sub("\*", "\*", ans_fixed)
        ans_fixed = re.sub("\+", "\+", ans_fixed)
        
        occurences_before = re.findall(ans_fixed, sentence_untok[:idx1])
        num_before = len(occurences_before)
            
        pattern1 = r'{}'.format("[\s▁]*".join(list(ans)))
        pattern2 = r'{}'.format("[\s▁]*".join(list(re.sub("[^\w\s]", "", ans))))

        pattern1 = re.sub("\?", "\?", pattern1)
        pattern1 = re.sub("\.", "\.", pattern1)
        pattern1 = re.sub("\+", "\+", pattern1)
        pattern2 = re.sub("\?", "\?", pattern2)
        pattern2 = re.sub("\.", "\.", pattern2)
        pattern2 = re.sub("\+", "\+", pattern2)

        ocurrences_tokenized = list(re.finditer(pattern1, sentence_tok))
        
        if len(ocurrences_tokenized) == 0:
            ocurrences_tokenized = list(re.finditer(pattern2, sentence_tok))
        
        match = ocurrences_tokenized[num_before]
        
        new_idx1 = match.span()[0]
        new_idx2 = match.span()[1]

        sentence_untok = sentence_untok.replace("Ç", "$")
        sentence_tok = sentence_tok.replace("Ç", "$")

        answer_indexes_tokenized.write(str(new_idx1) + ":" + str(new_idx2) + "\n")
        answer_tokenized.write(sentence_tok[new_idx1:new_idx2] + "\n")
        answer_untokenized.write(sentence_untok[idx1:idx2] + "\n")

    sentences_untokenized.close()
    sentences_tokenized.close()
    answer_indexes_untokenized.close()
    answer_indexes_tokenized.close()


process_answer_indexes("test.en", "test.32k.en", "answers-indexes.en.txt", "answers.en", "answers.32k.en.txt", "answers.en.txt")
