from pathlib import Path
from tokenizers.processors import BertProcessing
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
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


"""special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]

vocab_size = 30_522
max_length = 512
truncate_longer_samples = False
tokenizer = tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
)
tokenizer.train(files=paths, vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
)

# And then train
tokenizer.train(
    files=paths,
    vocab_size=30_522,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)


model_path = "Tokenizers/bert-tokenizer"

if not os.path.isdir(model_path):
  os.mkdir(model_path)

tokenizer.save_model(model_path)
with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)"""
  
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

model_path = "rbertT"

if not os.path.isdir(model_path):
  os.mkdir(model_path)
tokenizer.save_model(model_path)

tokenizer = RobertaTokenizer.from_pretrained(model_path)



"""tokenizer = ByteLevelBPETokenizer(
    "/home/nasim/working_dir/bertT-vocab.json",
    "/home/nasim/working_dir/bertT-merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
    ("<pad>", tokenizer.token_to_id("<pad>")),
    ("<mask>", tokenizer.token_to_id("<mask>")),
    ("<unk>", tokenizer.token_to_id("<unk>")),
)"""

tokens = tokenizer('Hello, how are you?')
print(
  tokenizer.encode("hello world!")
)
print(tokens)
# {'input_ids': [2, 21694, 16, 2287, 2009, 1991, 35, 3], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

#print(tokenizer.decode(tokens.tokens) )
# '[CLS] hello, how are you? [SEP]'