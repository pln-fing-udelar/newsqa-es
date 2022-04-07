import os
import random
import re
import csv

### This script is used to generate the following files (using the csv file newsqa_filtered.csv):
###   - test.en, test.es:   they contain the sentences of the corpus
###   - answers.en.txt:   it contains the answer in plain text for each of the sentences
###   - answers-indexes.en.txt:   it contains the index of the answer for each of the sentences

def extract_first_answer(text):
    first_answer = re.search(r"(.*?)##", text)
    if first_answer:
        return first_answer.group(1)
    else:
        return text

csv_input = open("newsqa_filtered.csv", "r", encoding="utf8")
csv_data = list(csv.reader(csv_input, delimiter=','))[1:]

output_src = open("test.en", "w", encoding="utf8")
output_tgt = open("test.es", "w", encoding="utf8")
output_ans_idx_en = open("answers-indexes.en.txt", "w", encoding="utf8")
output_ans_en = open("answers.en.txt", "w", encoding="utf8")

for entry in csv_data:
    indexes = re.search(r"\d+:\d+", entry[5])
    if indexes:
        ans_start = indexes.group(0).split(":")[0]
        ans_end = indexes.group(0).split(":")[1]
        if int(ans_start) > -1 and int(ans_end) > -1:
            output_ans_en.write(extract_first_answer(entry[2]) + "\n")
            output_ans_idx_en.write(str(ans_start) + ":" + str(ans_end) + "\n")
            output_src.write(extract_first_answer(entry[3]) + "\n")
            output_tgt.write(extract_first_answer(entry[4]) + "\n")


csv_input.close()
output_src.close()
output_tgt.close()
output_ans_idx_en.close()
output_ans_en.close()

