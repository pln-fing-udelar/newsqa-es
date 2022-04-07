import os
import random
import re
import csv

### This script is used to filter the rows with bad data in newsqa.csv, creating the newsqa_filtered.csv file

csv_input = open("newsqa.csv", "r", encoding="utf8")
csv_input2 = open("newsqa.csv", "r", encoding="utf8")
csv_data = list(csv.reader(csv_input, delimiter=','))
csv_lines = csv_input2.readlines()

csv_output = open("newsqa_filtered.csv", "w", encoding="utf8")
csv_output2 = open("newsqa_bad_rows.csv", "w", encoding="utf8")

csv_output.write(csv_lines[0])
csv_output2.write(csv_lines[0])

for idx, entry in enumerate(csv_data[1:]):
    ans_start = -1
    ans_end = -1
    indexes = re.search(r"\d+:\d+", entry[5])
    if indexes:
        ans_start = int(indexes.group(0).split(":")[0])
        ans_end = int(indexes.group(0).split(":")[1])

    if re.match(r"^(\s|\t|\n|\r)*$", str(entry[0])) == None and re.match(r"^(\s|\t|\n|\r)*$", str(entry[1])) == None and re.match(r"^(\s|\t|\n|\r)*$", str(entry[2])) == None and re.match(r"^(\s|\t|\n|\r)*$", str(entry[3])) == None and re.match(r"^(\s|\t|\n|\r)*$", str(entry[4])) == None and re.match(r"^(\s|\t|\n|\r)*$", str(entry[5])) == None and "*" not in str(entry[2]) and "Ã" not in str(entry[3]) and idx != 33676 and idx != 33677 and idx != 116925 and idx != 116926 and ans_start > -1 and ans_end > -1:
        csv_output.write(csv_lines[idx + 1])
    else:
        csv_output2.write(csv_lines[idx + 1])

csv_input.close()
csv_input2.close()
csv_output.close()
csv_output2.close()
