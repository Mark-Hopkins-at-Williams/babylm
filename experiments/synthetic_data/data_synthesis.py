from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
import random
from transformers import BertTokenizerFast, RobertaTokenizer
from collections import Counter
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("/mnt/storage/nasimb/babylm/all-base-guten-rarity-all-2p5k-rerun").to(torch_device)

def read_lines(filenames):
    for filename in filenames:
        with open(filename) as reader:
            n = 0
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

def create_multiple_files_dataset_dict(one_dataset):
    if one_dataset:
        corpora = ['wikipedia',]
    else:
        corpora = ['bnc_spoken', 'open_subtitles', 'aochildes', 
               'children_stories', 'cbt', 'gutenberg_fixed', 
               'qed', 'simple_wiki_mod', 'switchboard', 'wikipedia']
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
        load_from_cache_file=False
    )
    
else:
    raw_datasets = create_multiple_files_dataset_dict(Based_on_target_dataset)


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
    load_from_cache_file=False
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

#normalizing the counts
for token_id, count in token_counts.items():
    token_counts[token_id] = count / total_num_tokens
        
#calculate the order of raw sentences based on the rarity of the tokens in each dataset      
dict_ind_token_log_rarity = {i:sum([math.log(token_counts[token]) for token in list_tokenized_seqs[i]]) 
                         for i in range(len(list_tokenized_seqs))}

#calculate the order of raw sentences based on token length
tokenized_seq_lengths = [len(x) for x in tokenized_datasets["train"]["input_ids"]]
dict_ind_token_length = {i:tokenized_seq_lengths[i] for i in range(len(tokenized_seq_lengths))}


#rarity*******
sorted_indecies = sorted(dict_ind_token_rarity, key=lambda k:(dict_ind_token_rarity[k]))

#reorder the raw datatset
if Based_on_target_dataset:
    list_train_dataset_raw = list(raw_datasets["train"]["text"])
else:
    list_train_dataset_raw = list(raw_datasets_one["train"]["text"])

sorted_list_train_dataset_raw = [list_train_dataset_raw[i] for i in sorted_indecies]

#remove repeating instances from the list preserving the order, and cut
sorted_list_train_dataset_raw = list(dict.fromkeys(sorted_list_train_dataset_raw))

i = 1
with open('/mnt/storage/nasimb/babylm_data/wiki_syn_2.train', 'w') as f:
    while i < len(sorted_list_train_dataset_raw):
        #txt = "The museum features exhibits showcasing"
        mostf_input = sorted_list_train_dataset_raw[-i]
        token_len_mostf_input = dict_ind_token_length[sorted_indecies[-i]]
        start = mostf_input[:min(50, len(mostf_input)//2)].rsplit(" ", 1)[0]
        
        while start.count(" ") < 2:
            i += 1
            mostf_input = sorted_list_train_dataset_raw[-i]
            token_len_mostf_input = dict_ind_token_length[sorted_indecies[-i]]
            start = mostf_input[:min(50, len(mostf_input)//2)].rsplit(" ", 1)[0]
        
        model_inputs = tokenizer(start, return_tensors='pt').to(torch_device)

        output = model.generate(
        **model_inputs,
        max_new_tokens=128,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        no_repeat_ngram_size=2,
        min_new_tokens = max(5, len(model_inputs['input_ids'])*2/3)
        )
        
        res=tokenizer.decode(output[0], skip_special_tokens=True)
        #f.write(f"{i}: {start}\n")
        f.write(f"{res}\n")

        i += 1
        print(i, end = " ")
   

        
    


