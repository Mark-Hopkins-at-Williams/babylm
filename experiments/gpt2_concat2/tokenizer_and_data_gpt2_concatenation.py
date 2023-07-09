from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import random


def read_lines(filenames):
    for filename in filenames:
        with open(filename) as reader:
            for line in reader:
                line = line.strip()
                if len(line) > 0:
                    yield {'text': line}            

def create_dataset(filenames):
    return Dataset.from_generator(lambda: read_lines(filenames))

def create_dataset_dict(train_file_names, valid_file_names, test_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    result['valid'] = create_dataset(valid_file_names)
    result['test'] = create_dataset(test_file_names)
    return result


def create_multiple_files_dataset_dict():
    corpora = ['aochildes', 'open_subtitles', 'qed', 
               'switchboard', 'children_stories', 'bnc_spoken', 'simple_wikipedia',
               'wikipedia', 'cbt', 'gutenberg',]
    
    train_corpora = ['aochildes', 'bnc_spoken', 'open_subtitles',
               'children_stories', 'cbt_rarity_iorder_2k_p3k', 'gutenberg_fixed', 
               'qed', 'simple_wikipedia', 'switchboard', 'wikipedia']
    
    train_corpora = [f'../babylm_data/babylm_10M/{corpus}.train' for corpus in train_corpora]
    dev_corpora = [f'../babylm_data/babylm_dev/{corpus}.dev' for corpus in corpora]
    test_corpora = [f'../babylm_data/babylm_test/{corpus}.test' for corpus in corpora]
    return create_dataset_dict(train_corpora, dev_corpora, train_corpora)


CONTEXT_LENGTH = 128
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def tokenize(element):
    outputs = TOKENIZER(element["text"], truncation=False)
    input_batch = []
    next_segment = []
    
    for input_ids in outputs["input_ids"]:
        next_segment.extend(input_ids)
        next_segment.append(TOKENIZER.eos_token_id)
        while len(next_segment) >= CONTEXT_LENGTH:
            input_batch.append(next_segment[:CONTEXT_LENGTH])
            next_segment = next_segment[CONTEXT_LENGTH:]
    return {"input_ids": input_batch}
    

