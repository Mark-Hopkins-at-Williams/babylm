from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
from DP_merging import dp_merge_inputs
import random
from transformers import BertTokenizerFast, RobertaTokenizer

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

#calculate the order of raw sentences based on token length
tokenized_seq_lengths = [len(x) for x in tokenized_datasets["train"]["input_ids"]]
dict_ind_token_length = {i:tokenized_seq_lengths[i] for i in range(len(tokenized_seq_lengths))}
sorted_indecies = sorted(dict_ind_token_length, key=lambda k:dict_ind_token_length[k])

#reorder the raw datatset
list_train_dataset_raw = list(raw_datasets["train"]["text"])
sorted_list_train_dataset_raw = [list_train_dataset_raw[i] for i in sorted_indecies]

#write the reordered dataset to a new file
with open('train_data_cl_length.txt', 'w') as f:
    for sent in sorted_list_train_dataset_raw:
        f.write(f"{sent}\n")
        

