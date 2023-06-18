from pathlib import Path
from tokenizers.processors import BertProcessing
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


tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

model_path = "rbertT"

if not os.path.isdir(model_path):
  os.mkdir(model_path)
tokenizer.save_model(model_path)

tokenizer = RobertaTokenizer.from_pretrained(model_path)


"""tokens = tokenizer('Hello, how are you?')
print(
  tokenizer.encode("hello world!")
)
print(tokens)"""
