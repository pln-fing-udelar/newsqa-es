## This is a guide to generate the aligned answers in spanish, using the Mask-Align algorithm

0. Train Mask-Align and get the newsqa.csv corpus file

1. To remove rows with unanswered questions, move to the directory run the remove_bad_rows.py script in a folder with the corpus file.

2. To generate test.en, test.es, answers-indexes.en.txt and answers.en.txt run in the same folder the script generate_files.py.

3. Run the following commands using the SentencePiece models learned during the training of Mask-Align:  
    spm_encode --model=en.model --output_format=piece < test.en > test.32k.en  
    spm_encode --model=es.model --output_format=piece < test.es > test.32k.es  

4. Run python process_answer_indexes.py

5. Create a folder named corpus-es in the root folder of the Mask-Align repo, and copy the files test.32k.en, test.32k.es y answers.en in it

6. In the root folder, run mask-align: bash thualign/bin/generate.sh -s spanish -o output.txt

7. The file output.txt has the answers marked in the spanish sentences with the characters "{{" and "}}". The file is tokenized, so we must decode it.

8. Decode output.txt:  
    spm_decode --model=es.model --input_format=piece < output.txt > output-plain.txt

9. Run the script output_brackets_to_indexes.py in the folder where the file output-plain.txt is located 

The following three files are generated:  
- output-indexes.txt: It contains the indexes of the answers in spanish  
- output-answers.txt: It contains the answers in spanish (in plain text)  
- output-sentences.txt: It contains the sentences in spanish (untokenized)
