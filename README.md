# newsqa-esp

This is a guide to translate the NewsQA question answering corpus to spanish.

The first step is to translate every sentence and question from the corpus. After doing that, for each answer of the english corpus we need to find which substring in the translated sentence corresponds to the answer. To achieve this we use a word alignment algorithm named Mask-Align.


The repo is organized in the following three folders:

  1) **preprocess-corpus:**  explains how to preprocess the NewsQA corpus to generate a csv file in the format needed 
  2) **train-mask-align:**  explains how to train and set up Mask-Align to generate the alignments to get the answers in spanish 
  3) **generate-alignments:**  explains how to use Mask-Align to generate the alignments 
