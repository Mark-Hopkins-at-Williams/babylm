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

def create_multiple_files_dataset_dict():
    corpora = ['aochildes', 'bnc_spoken', 'open_subtitles',
               'children_stories', 'cbt', 'gutenberg_fixed', 
               'qed', 'simple_wikipedia', 'switchboard', 'wikipedia']
    train_corpora = [f'/mnt/storage/nasimb/babylm_data/babylm_10M/{corpus}.train' for corpus in corpora]
    return create_dataset_dict(train_corpora)
      

CONTEXT_LENGTH = 128

class Gpt2Parameters:
    model_arch = "gpt2"
    is_mlm = False
    explicit_bos_token = True
    explicit_eos_token = True
    pad_token = '[PAD]'
    context_length = CONTEXT_LENGTH
    
params = Gpt2Parameters()
TOKENIZER = AutoTokenizer.from_pretrained(params.model_arch)

def tokenize(element):
    outputs = TOKENIZER(element["text"], truncation=False)  
    return {"input_ids": outputs["input_ids"]}


#create raw train dataset and tokenize ir
raw_datasets = create_multiple_files_dataset_dict()
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
  
"""#calculate the order of raw sentences based on the rarity of the tokens in each dataset      
dict_ind_token_rarity = {i:(sum([token_counts[token] for token in list_tokenized_seqs[i]]) 
                            / len(list_tokenized_seqs[i])) 
                         for i in range(len(list_tokenized_seqs))}
max_rarity = max(list(dict_ind_token_rarity.values()))"""


#normalizing the counts
normalized_token_counts = Counter()
for token_id, count in token_counts.items():
    normalized_token_counts[token_id] = count / total_num_tokens
        
#calculate the order of raw sentences based on the rarity of the tokens in each dataset      
dict_ind_token_log_rarity = {i: -1 * sum([math.log(normalized_token_counts[token]) for token in list_tokenized_seqs[i]]) 
                         for i in range(len(list_tokenized_seqs))}
max_log_rarity = max(list(dict_ind_token_log_rarity.values()))

#calculate the order of raw sentences based on token length
tokenized_seq_lengths = [len(x) for x in tokenized_datasets["train"]["input_ids"]]
dict_ind_token_length = {i:tokenized_seq_lengths[i] for i in range(len(tokenized_seq_lengths))}

#soritng based log rarity!!
sorted_indecies = sorted(dict_ind_token_log_rarity, 
                         key=lambda k:dict_ind_token_log_rarity[k])

#reorder the raw datatset
list_train_dataset_raw = list(raw_datasets["train"]["text"])
sorted_list_train_dataset_raw = [list_train_dataset_raw[i] for i in sorted_indecies]

#remove repeating instances from the list preserving the order
final_sorted_list_train_dataset_raw = list(dict.fromkeys(sorted_list_train_dataset_raw))

batch_size = 32
t_competent = 220000 
c0_squared = (1/t_competent)**2
num_sent = len(final_sorted_list_train_dataset_raw)
total_tokens = 0
random.seed(1)
file_name = 'train_data_cl_log_rarity_220k.txt'
with open(file_name, 'w') as f:
    for t in tqdm(range(t_competent)):
        c_sqrt = min(1, math.sqrt(t * ((1 - c0_squared) / t_competent) + c0_squared))
        max_ind = max(batch_size, int(c_sqrt * num_sent))
        sample_inds = np.random.randint(low = 0,high=max_ind,size=batch_size)
        for ind in sample_inds:
            f.write(f"{final_sorted_list_train_dataset_raw[ind]}\n")
        
    f.close()


def create_final_file_dataset_dict():
    train_corpora = [file_name]
    return create_dataset_dict(train_corpora)        
raw_datasets = create_final_file_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
    load_from_cache_file=False
)

#count number of tokens in the train dataset
list_tokenized_seqs = list(tokenized_datasets["train"]["input_ids"])
total_num_tokens = 0
for seq in list_tokenized_seqs:
    total_num_tokens += len(seq)
        
print(total_num_tokens)
