# NewsQA-es

NewsQA-es is a Spanish version of [the NewsQA Dataset](https://www.microsoft.com/en-us/research/project/newsqa-dataset/)
, created by researchers at [Grupo PLN, UdelaR](https://www.fing.edu.uy/inco/grupos/pln/).

## Obtaining the dataset

Due to license issues, we can't provide a download link. Therefore, here we provide the steps to re-create it by
translating NewsQA. The steps:

1. Download the NewsQA dataset. Follow the steps in the
   [NewsQA website](https://www.microsoft.com/en-us/research/project/newsqa-dataset/) to download the dataset.
2. Obtain the answers text with the tools from [Maluuba NewsQA](https://github.com/Maluuba/newsqa).
3. Translate every sentence and question. Follow the steps described in the next section.
4. Use a translation aligner to find the correspondence between each answer from NewsQA and a span of text from the
   translated sentence in Spanish. Follow the steps in the repo
   [pln-fing-udelar/Mask-Align](https://github.com/pln-fing-udelar/Mask-Align).

## Translating NewsQA into Spanish

We translated the dataset using the [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) model from Helsinki-NLP.
1. Download the translate.py script to the same directory as cnn_stories folder is.
2. Execute translate.py
3. You will find the translated stories into cnn_stories/cnn/translated folder.

## Contact Us

If you encounter issues following these steps,
please [open a GitHub issue](https://github.com/pln-fing-udelar/newsqa-es/issues/new)
or [email us at pln@fing.edu.uy](mailto:pln@fing.edu.uy).
