from pathlib import Path
from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
from transformers import PreTrainedTokenizerFast

paths = ["../babylm_data/babylm_10M/switchboard.train",
                                "../babylm_data/babylm_10M/qed.train", 
                                "../babylm_data/babylm_10M/open_subtitles.train", 
                                "../babylm_data/babylm_10M/wikipedia.train", 
                                "../babylm_data/babylm_10M/bnc_spoken.train", 
                                "../babylm_data/babylm_10M/cbt.train", 
                                "../babylm_data/babylm_10M/children_stories.train", 
                                "../babylm_data/babylm_10M/gutenberg.train", 
                                "../babylm_data/babylm_10M/simple_wikipedia.train"]

"""tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])"""

tokenizer = ByteLevelBPETokenizer()
# and train
tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
tokenizer.save_model('tokenizer')
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('tokenizer')

"""tokenizer = BertWordPieceTokenizer(
         clean_text=True,
        strip_accents=False,
        lowercase=True
)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2,special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]', 'cls_token': '[CLS]', 'sep_token':'[SEP]', 
                              'unk_token': '[UNK]'})

# Save files to disk

tokenizer.save("./distillbert-tokenizer")

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/nasim/working_dir/distillbert-tokenizer")
"""

print(
    tokenizer.encode("hello world!")
)
tokens = tokenizer('Hello, how are you?')
print(tokens)
# {'input_ids': [2, 21694, 16, 2287, 2009, 1991, 35, 3], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

print(tokenizer.decode(tokens["input_ids"]) )
# '[CLS] hello, how are you? [SEP]'