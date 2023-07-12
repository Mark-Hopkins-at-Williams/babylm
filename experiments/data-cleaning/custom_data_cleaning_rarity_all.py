from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
import random
from transformers import BertTokenizerFast, RobertaTokenizer
from collections import Counter
import math
import numpy as np
from tqdm import tqdm

def read_lines(filenames):
    for filename in filenames:
        with open(filename) as reader:
            for line in reader:
                line = line.strip()
                if len(line) > 0:
                    yield {'text': line}   
                
                
def create_dataset(filenames):
    return Dataset.from_generator(lambda: read_lines(filenames))

def create_dataset_dict(train_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    return result

def create_multiple_files_dataset_dict(one_dataset, corpus=None):
    if one_dataset:
        corpora = [corpus]
    else:
        corpora = ['bnc_spoken', 'open_subtitles', 'aochildes', 
               'children_stories', 'cbt', 'gutenberg_fixed', 
               'qed', 'simple_wikipedia', 'switchboard', 'wikipedia']
    print(corpora)
    train_corpora = [f'/mnt/storage/nasimb/babylm_data/babylm_10M/{corpus}.train' for corpus in corpora]
    return create_dataset_dict(train_corpora)
      

CONTEXT_LENGTH = 128
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

def tokenize(element):
    outputs = TOKENIZER(element["text"], truncation=False)  
    return {"input_ids": outputs["input_ids"]}


raw_datasets = create_multiple_files_dataset_dict(False)
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
    #load_from_cache_file=False
)

#count number of tokens in the train dataset
list_tokenized_seqs = list(tokenized_datasets["train"]["input_ids"])
token_counts = Counter()
total_num_tokens = 0
for seq in list_tokenized_seqs:
    for token in seq:
        token_counts[token] += 1
        total_num_tokens += 1

corpora = ['aochildes', 'open_subtitles', 'qed', 'switchboard',
           'children_stories', 'bnc_spoken', 'simple_wikipedia',
           'wikipedia', 'cbt','gutenberg_fixed'] 

for i in range(10):       
    raw_datasets_one = create_multiple_files_dataset_dict(True, corpora[i])
    tokenized_datasets_one = raw_datasets_one.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
        #load_from_cache_file=False
    )        
    list_tokenized_seqs = list(tokenized_datasets_one["train"]["input_ids"])

    #calculate the order of raw sentences based on the rarity of the tokens in each dataset      
    dict_ind_token_rarity = {i:(sum([token_counts[token] for token in list_tokenized_seqs[i]]) 
                                / len(list_tokenized_seqs[i])) 
                            for i in range(len(list_tokenized_seqs))}
            

    sorted_indecies = sorted(dict_ind_token_rarity, key=lambda k:(dict_ind_token_rarity[k]))

    list_train_dataset_raw = list(raw_datasets_one["train"]["text"])

    train_dataset_raw_cleaned = [list_train_dataset_raw[i] for i in sorted_indecies]

    print(len(train_dataset_raw_cleaned))
    train_dataset_raw_cleaned = list(dict.fromkeys(train_dataset_raw_cleaned))

    with open(f'/mnt/storage/nasimb/babylm_data/babylm_10M/rarity_all/{corpora[i]}.train', 'w') as f:
        for sent in train_dataset_raw_cleaned:
            f.write(f"{sent}\n")
    


