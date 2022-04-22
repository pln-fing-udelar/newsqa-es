from transformers import MarianTokenizer, MarianMTModel
from os import listdir
from os.path import isfile, join

def chunks(lst, n):
    # Yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

path_stories = "cnn_stories/cnn/stories/"
path_translated = "cnn_stories/cnn/translated/"
model_name = "Helsinki-NLP/opus-mt-en-es"
# Download the model and the tokenizer
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


# Files
onlyfiles = [f for f in listdir(path_stories) if isfile(join(path_stories, f))]
for file_news in onlyfiles:
	try:
		file_story = open(path_stories + file_news, 'r')
		file_translated = open(path_translated + file_news, "w")
		text = file_story.readlines()
		lines = []
		for line in text:
			if line != '\n':
				lines.append(line)
		for ck in chunks(lines, 20):
			# Tokenize the text
			batch = tokenizer(ck, return_tensors="pt", padding=True)
			batch["input_ids"] = batch["input_ids"][:, :512]
			batch["attention_mask"] = batch["attention_mask"][:, :512]
			# Translate
			translation = model.generate(**batch)
			out_text = tokenizer.batch_decode(translation, skip_special_tokens=True)

			for out_line in out_text:
				file_translated.write(out_line + '\n')
		file_translated.close()
		file_story.close()
	except:
		print(file_news)
		raise
		

