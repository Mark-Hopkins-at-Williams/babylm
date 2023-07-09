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
                bar_count =  line.count('|')
                exclamation_count = line.count('!')
                line = line.replace("= = =", "")
                if len(line) > 0 and line[0] == "-":
                    line = line[1:]
                if len(line) > 0 and bar_count < 2 and bar_count + exclamation_count < 2:
                    yield {'text': line}   
                
                
def create_dataset(filenames):
    return Dataset.from_generator(lambda: read_lines(filenames))

def create_dataset_dict(train_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    return result

def create_multiple_files_dataset_dict(one_dataset):
    if one_dataset:
        #datasets ordered based on a manual difficulty evaluaiton
        corpora = ['aochildes_modified_length_14k', 'open_subtitles', 'qed', 
               'switchboard', 'children_stories', 'bnc_spoken', 'simple_wikipedia',
               'wikipedia', 'cbt_modified_rarity_2k_.3k', 'gutenberg_modified_rarity_2.2k_1k',]
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


Based_on_target_dataset = True

if not Based_on_target_dataset:

    raw_datasets = create_multiple_files_dataset_dict(Based_on_target_dataset)
    
    raw_datasets_one = create_multiple_files_dataset_dict(not Based_on_target_dataset)
    tokenized_datasets_one = raw_datasets_one.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
        #load_from_cache_file=False
    )
    
else:
    raw_datasets = create_multiple_files_dataset_dict(Based_on_target_dataset)


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
        
        
if not Based_on_target_dataset:
    list_tokenized_seqs = list(tokenized_datasets_one["train"]["input_ids"])

#calculate the order of raw sentences based on the rarity of the tokens in each dataset      
dict_ind_token_rarity = {i:(sum([token_counts[token] for token in list_tokenized_seqs[i]]) 
                            / len(list_tokenized_seqs[i])) 
                         for i in range(len(list_tokenized_seqs))}
        
"""#normalizing the counts
for token_id, count in token_counts.items():
    token_counts[token_id] = count / total_num_tokens

        
#calculate the order of raw sentences based on the rarity of the tokens in each dataset      
dict_ind_token_log_rarity = {i:sum([math.log(token_counts[token]) for token in list_tokenized_seqs[i]]) 
                         for i in range(len(list_tokenized_seqs))}
"""

#calculate the order of raw sentences based on token length
tokenized_seq_lengths = [len(x) for x in tokenized_datasets["train"]["input_ids"]]
dict_ind_token_length = {i:tokenized_seq_lengths[i] for i in range(len(tokenized_seq_lengths))}

sorted_indecies = sorted(dict_ind_token_rarity, key=lambda k:(dict_ind_token_rarity[k]))

#reorder the raw datatset
if Based_on_target_dataset:
    list_train_dataset_raw = list(raw_datasets["train"]["text"])
else:
    list_train_dataset_raw = list(raw_datasets_one["train"]["text"])


#theory: preservig the order of setences in a dataset matters
metric_sorted_indicies_cut = sorted_indecies#[16000:859500]

train_dataset_raw_cleaned = [list_train_dataset_raw[i] for i in sorted(metric_sorted_indicies_cut)]

#remove repeating instances from the list preserving the order => after the cut indivies are known 
#train_dataset_raw_cleaned = list(dict.fromkeys(train_dataset_raw_cleaned))

with open('/mnt/storage/nasimb/babylm_data/babylm_10M/all_mod_datasets_rarity_all_iorder_no_cut_repetition.train', 'w') as f:
    for sent in train_dataset_raw_cleaned:
        f.write(f"{sent}\n")
    


