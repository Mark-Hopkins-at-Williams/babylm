from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
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

def create_dataset_dict(train_file_names, valid_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    result['valid'] = create_dataset(valid_file_names)
    return result

def create_multiple_files_dataset_dict():
    corpora = ['aochildes', 'children_stories',
               'bnc_spoken', 'cbt', 'gutenberg',
               'open_subtitles', 'qed', 'simple_wikipedia',
               'switchboard', 'wikipedia']
    train_corpora = [f'../babylm_data/babylm_10M/{corpus}.train' for corpus in corpora]
    dev_corpora = [f'../babylm_data/babylm_dev/{corpus}.dev' for corpus in corpora]
    #test_corpora = [f'../babylm_data/babylm_test/{corpus}.test' for corpus in corpora]
    return create_dataset_dict(train_corpora, dev_corpora)
    
    

CONTEXT_LENGTH = 128
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize(element):
    outputs = TOKENIZER(element["text"], truncation=True, padding="max_length",
                   max_length=CONTEXT_LENGTH, return_special_tokens_mask=True)
    return outputs
    


raw_datasets = create_multiple_files_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
