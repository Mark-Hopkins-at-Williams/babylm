from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
from DP_merging import dp_merge_inputs
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
                bar_count =  line.count('|')
                exclamation_count = line.count('!')
                line = line.replace("= = =", "")
                if len(line) > 0 and line[0] == "-":
                    line = line[1:]
                if len(line) > 0 and bar_count < 2 and bar_count + exclamation_count < 2:
                    yield {'text': line}   
                """
                #gutenberg
                line = line.strip()
                bar_count =  line.count('|')
                exclamation_count = line.count('!')
                line = line.replace("= = =", "")
                gutenberg_line += line
                if len(line) == 0 and len(gutenberg_line) > 1:
                    yield {'text': gutenberg_line}         
                    gutenberg_line = ""  """        

def create_dataset(filenames):
    return Dataset.from_generator(lambda: read_lines(filenames))

def create_dataset_dict(train_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    return result

def create_multiple_files_dataset_dict():
    corpora = ['cbt']
    train_corpora = [f'/mnt/storage/nasimb/babylm_data/babylm_10M/{corpus}.train' for corpus in corpora]
    return create_dataset_dict(train_corpora)
      

CONTEXT_LENGTH = 128
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

def tokenize(element):
    outputs = TOKENIZER(element["text"], truncation=False)  
    return {"input_ids": outputs["input_ids"]}


#create raw train dataset and tokenize ir
raw_datasets = create_multiple_files_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
    load_from_cache_file=False
)

#count number of tokens in the train dataset
list_tokenized_seqs = list(tokenized_datasets["train"]["input_ids"])
token_counts = Counter()
for seq in list_tokenized_seqs:
    for token in seq:
        token_counts[token] += 1
  
#calculate the order of raw sentences based on the rarity of the tokens in each dataset      
dict_ind_token_rarity = {i:(sum([token_counts[token] for token in list_tokenized_seqs[i]]) 
                            / len(list_tokenized_seqs[i])) 
                         for i in range(len(list_tokenized_seqs))}

#calculate the order of raw sentences based on token length
tokenized_seq_lengths = [len(x) for x in tokenized_datasets["train"]["input_ids"]]
dict_ind_token_length = {i:tokenized_seq_lengths[i] for i in range(len(tokenized_seq_lengths))}


#print({k: dict_ind_token_length[k] for k in list(dict_ind_token_length.keys())[:100]})
#print({k: dict_ind_token_rarity[k] for k in list(dict_ind_token_rarity.keys())[:100]})

sorted_indecies = sorted(dict_ind_token_rarity, key=lambda k:(dict_ind_token_rarity[k]))

#reorder the raw datatset
list_train_dataset_raw = list(raw_datasets["train"]["text"])
sorted_list_train_dataset_raw = [list_train_dataset_raw[i] for i in sorted_indecies]

#remove repeating instances from the list preserving the order, and cut
sorted_list_train_dataset_raw = list(dict.fromkeys(sorted_list_train_dataset_raw))

with open('/mnt/storage/nasimb/babylm_data/babylm_10M/cbt_modified.train', 'w') as f:
    for sent in sorted_list_train_dataset_raw:
        f.write(f"{sent}\n")
    


