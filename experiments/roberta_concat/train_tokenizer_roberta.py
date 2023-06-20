from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
import os
import json

paths = ["../babylm_data/babylm_10M/switchboard.train",
                                "../babylm_data/babylm_10M/qed.train", 
                                "../babylm_data/babylm_10M/open_subtitles.train", 
                                "../babylm_data/babylm_10M/wikipedia.train", 
                                "../babylm_data/babylm_10M/bnc_spoken.train", 
                                "../babylm_data/babylm_10M/cbt.train", 
                                "../babylm_data/babylm_10M/children_stories.train", 
                                "../babylm_data/babylm_10M/gutenberg.train", 
                                "../babylm_data/babylm_10M/simple_wikipedia.train"]


tokenizer = ByteLevelBPETokenizer(lowercase=True)
tokenizer.train(files=paths, vocab_size=8192, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>",
])

model_path = "RobertaTokenizer"

if not os.path.isdir(model_path):
  os.mkdir(model_path)
tokenizer.save_model(model_path)

tokenizer = RobertaTokenizer.from_pretrained(model_path)


"""tokens = tokenizer('Hello, how are you?')
print(
  tokenizer.encode("hello world!")
)
print(tokens)"""
