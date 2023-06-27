from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
from DP_merging import dp_merge_inputs
import random
from transformers import BertTokenizerFast, RobertaTokenizer
from collections import Counter


def read_lines(filenames):
    for filename in filenames:
        with open(filename) as reader:
            for line in reader:
                line = line.strip()
                if len(line) > 0:
                    yield {'text': line}            

def create_dataset(filenames):
    return Dataset.from_generator(lambda: read_lines(filenames))

def create_dataset_dict(train_file_names, valid_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    result['valid'] = create_dataset(valid_file_names)
    return result

def create_multiple_files_dataset_dict():
    corpora = ['aochildes', 'open_subtitles', 'qed', 
               'switchboard', 'children_stories', 'bnc_spoken',
               'wikipedia', 'cbt', 'gutenberg',]
    train_corpora = [f'../babylm_data/babylm_10M/{corpus}.train' for corpus in corpora]
    dev_corpora = [f'../babylm_data/babylm_dev/{corpus}.dev' for corpus in corpora]
    #test_corpora = [f'../babylm_data/babylm_test/{corpus}.test' for corpus in corpora]
    return create_dataset_dict(train_corpora, dev_corpora)
    
    

CONTEXT_LENGTH = 128
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def tokenize(dataset):
    outputs = [TOKENIZER(element["text"], truncation=False)['input_ids'] for element in dataset]
    print("finished tokenizing")
    
    token_counts = Counter()
    for element in outputs:
        for token in element:
            token_counts[token] += 1
    print("finished counting tokens")
    
    outputs_copy = list(outputs)
    outputs_sorted = sorted(outputs, key=lambda x: (sum([token_counts[token] for token in x]) / len(x), outputs_copy.index(x)))
    print("finished soering the sequences based on rarity")
    
    merged_inputs = dp_merge_inputs(outputs_sorted, CONTEXT_LENGTH, TOKENIZER.eos_token_id)
    print("finished merging the tokenized sequences with dp")

    return {"input_ids": merged_inputs}

