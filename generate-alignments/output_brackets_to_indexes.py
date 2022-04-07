import os
import re
import random

input_file = open("./output-plain.txt", "r", encoding="utf-8")
output_file = open("./output-indexes.txt", "w", encoding="utf-8")
answer_file = open("./output-answers.txt", "w", encoding="utf-8")
sentence_file = open("./output-sentences.txt", "w", encoding="utf-8")

lines = input_file.readlines()

for line in lines:
    match1 = re.search("{{", line)
    match2 = re.search("}}", line)
    if match1 and match2:
        start_idx = match1.span()[0]
        end_idx = match2.span()[0] - 2
        line = re.sub(r"{{", "", line)
        line = re.sub(r"}}", "", line)

        while line[start_idx] == " ":
            start_idx += 1
        
        output_file.write(str(start_idx) + ":" + str(end_idx) + "\n")
        answer_file.write(line[start_idx:end_idx] + "\n")
        sentence_file.write(line)
    else:
        output_file.write("\n")
        answer_file.write("\n")
        sentence_file.write(line)