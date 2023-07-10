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
            #gutenberg_line = ""
            for line in reader:
                line = line.strip()
                if len(line) > 0:
                    yield {'text': line}   
                
                """#gutenberg
                line = line.strip()
                if len(gutenberg_line) > 1:
                    gutenberg_line = gutenberg_line + " " + line
                else:
                    gutenberg_line += line
                if len(line) == 0 and len(gutenberg_line) > 1:
                    yield {'text': gutenberg_line}    
                    gutenberg_line = "" """

def create_dataset(filenames):
    return Dataset.from_generator(lambda: read_lines(filenames))

def create_dataset_dict(train_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    return result

def create_multiple_files_dataset_dict(one_dataset):
    if one_dataset:
        corpora = ['gutenberg_fixed']
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


Based_on_target_dataset = False

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
rarity_order = True
#sorted_indecies = sorted(sorted_indecies)

sorted_list_train_dataset_raw = [list_train_dataset_raw[i] for i in sorted_indecies]
   
sorted_list_train_dataset_raw_final = []
n = 1
for i in range(len(sorted_list_train_dataset_raw)):
    sent = sorted_list_train_dataset_raw[i]
    num_count = sum(c.isdigit() for c in sent)
    comma_count = sum(c == ',' for c in sent)
    if (not (len(sent) < 25 and i > 20000 and rarity_order)) and (not (num_count > 2 and (comma_count > 1 or sent[-1].isdigit() or sent[-2].isdigit() or sent[-3].isdigit()) and len(sent) < 150)):
        sorted_list_train_dataset_raw_final.append(sent)       
    else:
        #print(sent)
        n += 1
        
print(n)
sorted_list_train_dataset_raw = sorted_list_train_dataset_raw_final

#remove repeating instances from the list preserving the order, and cut
sorted_list_train_dataset_raw = list(dict.fromkeys(sorted_list_train_dataset_raw))[4000:20000]

with open('/mnt/storage/nasimb/babylm_data/babylm_10M/guten_mod_rarity_all_4k_p12k.train', 'w') as f:
    for sent in sorted_list_train_dataset_raw:
        f.write(f"{sent}\n")
        
    


