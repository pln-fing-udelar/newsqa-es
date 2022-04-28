# NewsQA-es

NewsQA-es is a Spanish version of
[the NewsQA Dataset](https://www.microsoft.com/en-us/research/project/newsqa-dataset/), created by researchers at
[Grupo PLN, UdelaR](https://www.fing.edu.uy/inco/grupos/pln/).

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

We translated the dataset using the [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) model from Helsinki-NLP. To 
reproduce it (having already downloaded the NewsQA dataset):

1. Clone this repo:

   ```bash
   git clone https://github.com/pln-fing-udelar/newsqa-es
   cd newsqa-es/
   ```

2. Set up the environment using [Conda](https://docs.conda.io/en/latest/index.html):

   ```bash
   conda env create
   conda activate newsqa-es
   ```

3. Place the extracted CNN stories from the NewsQA dataset under `cnn_stories/cnn/stories`:

   ```bash
   mkdir cnn_stories
   tar -xvf cnn_stories.tgz -C cnn_stories/
   ```

4. Run the following command to translate the dataset. Consider that it takes time, and you may benefit from having a 
   GPU. For reference, it takes a bit less than a day and a half on a computer with an Nvidia RTX 2080 Ti GPU. 
   Consider changing the `BATCH_SIZE` constant to best fit your hardware (with a value that's too high you may 
   incur in OOM errors; if it's too low you are underutilizing your resources, and it could be faster).

   ```bash
   mkdir -p cnn_stories/cnn/translated
   ./translate.py
   ```

5. You will find the translated stories under the folder `cnn_stories/cnn/translated/`.

TODO: how to go from these files to the `newsqa.csv` file required in Mask-Align?

## Contact Us

If you encounter issues following these steps,
please [open a GitHub issue](https://github.com/pln-fing-udelar/newsqa-es/issues/new)
or [email us at pln@fing.edu.uy](mailto:pln@fing.edu.uy).
