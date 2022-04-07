## This is a guide to train Mask-Align and set it up to align the answers of the NewsQA corpus

### Install requirements and download files
  - Python 3.9
  - CUDA Driver
  - Google Sentencepiece
  - Install pip with the following requirements
    - tensorFlow==2.5.0
    - six==1.15.0
    - regex==2021.7.6
    - numpy==1.19.5
    - torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    - nltk==3.6.2

Clone Mask-Align from this url:  https://github.com/THUNLP-MT/Mask-Align

From our repo, download the following files:
  - spanish.config: copy it to the folder thualign/configs/user of the Mask-Align repo
  - infer_fast.py and generate.sh: copy them to the folder thualign/bin of the Mask-Align repo  


### Prepare a corpus (this is an example for the europarl corpus)
- Download from https://www.statmt.org/europarl/
- Remove sentences that don't have a pair (sentences that correspond with an empty line)
- Remove sencences of length 1
- Remove sentences that contain tags (characters "<" and ">")
- Split the corpus in test, train and val. A good size can be 500 sentences for test and 2000 for val.
- Name the files corpus.es, validacion.es, test.es, corpus.en, validacion.en and test.en
- Run the following commands, to learn the vocabulary and tokenize the files:

        - spm_train --input=corpus.en --model_prefix=en --vocab_size=32000 --character_coverage=1.0 --model_type=unigram
        - spm_train --input=corpus.es --model_prefix=es --vocab_size=32000 --character_coverage=1.0 --model_type=unigram
        - spm_encode --model=en.model --output_format=piece < corpus.en > corpus.32k.en
        - spm_encode --model=en.model --output_format=piece < validation.en > validation.32k.en
        - spm_encode --model=en.model --output_format=piece < test.en > test.32k.en
        - spm_encode --model=es.model --output_format=piece < corpus.es > corpus.32k.es
        - spm_encode --model=es.model --output_format=piece < validation.es > validation.32k.es
        - spm_encode --model=es.model --output_format=piece < test.es > test.32k.es

- Run: `python3 thualign/scripts/shuffle_corpus.py --corpus corpus.32k.es corpus.32k.en`
- In the vocabulary files, replace the tags \<s\> and \</s\> for \<eos\> and \<pad\>

### Train the model
 - In the config file thualign/configs/user/spanish.config, specify the location of the following files:
    - corpus.32k.es.shuf
    - corpus.32k.en.shuf
    - validation.32k.es
    - validation.32k.en
    - test.32k.es
    - test.32k.en
    - es.vocab
    - en.vocab
  - In `device_list` specify the number of GPUs
  - In `batch_size` choose the highest value that doesn't make the training stop due to lack of memory (try different numbers)
  - The value of `update_cycle` must be 36000 / batch_size
  - To train, run: `bash thualign/bin/train.sh -s spanish`
  - The model is saved in a folder created in the root directory of the repository
