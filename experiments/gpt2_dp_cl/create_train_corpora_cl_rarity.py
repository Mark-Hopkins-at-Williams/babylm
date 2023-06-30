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

def create_dataset(filenames):
    return Dataset.from_generator(lambda: read_lines(filenames))

def create_dataset_dict(train_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    return result

def create_multiple_files_dataset_dict():
    corpora = ['aochildes', 'bnc_spoken', 'open_subtitles',
               'children_stories', 'cbt', 'gutenberg', 
               'qed', 'simple_wikipedia', 'switchboard', 'wikipedia']
    train_corpora = [f'../babylm_data/babylm_10M/{corpus}.train' for corpus in corpora]
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
sorted_indecies = sorted(dict_ind_token_rarity, key=lambda k:dict_ind_token_rarity[k], reverse=True)

#reorder the raw datatset
list_train_dataset_raw = list(raw_datasets["train"]["text"])
sorted_list_train_dataset_raw = [list_train_dataset_raw[i] for i in sorted_indecies]

#remove repeating instances from the list preserving the order
sorted_list_train_dataset_raw = list(dict.fromkeys(sorted_list_train_dataset_raw))
    
#no cuts in the rarity ordered dataset

#sampling with the square root function
batch_size = 32
t_competent = int((800000/32)*5) #5 epochs to competence, 800000 sentences in batches of 32 
c0_squared = (1/t_competent)**2
num_sent = len(sorted_list_train_dataset_raw)
with open('train_data_rarity_cl_sampling.txt', 'w') as f:
    for t in tqdm(range(t_competent)):
        c_sqrt = min(1, math.sqrt(t * ((1 - c0_squared) / t_competent) + c0_squared))
        max_ind = max(batch_size, int(c_sqrt * num_sent))
        sample_inds = np.random.randint(low = 0,high=max_ind,size=batch_size)
        for ind in sample_inds:
            f.write(f"{sorted_list_train_dataset_raw[ind]}\n")
        

