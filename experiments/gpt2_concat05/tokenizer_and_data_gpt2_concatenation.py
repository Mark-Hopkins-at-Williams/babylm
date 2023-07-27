from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import random


def read_lines(filenames, bos_tok=None, eos_tok=None):
    for filename in filenames:
        with open(filename) as reader:
            for line in reader:
                line = line.strip()                
                if len(line) > 0:
                    if bos_tok is not None:
                        line = f"{bos_tok} {line}"
                    if eos_tok is not None:
                        line = f"{line} {eos_tok}"
                    yield {'text': line}           

def create_dataset(filenames, bos_tok, eos_tok):
    return Dataset.from_generator(lambda: read_lines(filenames, bos_tok, eos_tok))


def create_dataset_dict(train_files, valid_files, test_files, bos_tok, eos_tok):
    result = DatasetDict()
    result['train'] = create_dataset(train_files, bos_tok, eos_tok)
    result['valid'] = create_dataset(valid_files, bos_tok, eos_tok)
    result['test'] = create_dataset(test_files, bos_tok, eos_tok)
    return result


def create_multiple_files_dataset_dict(bos_tok=None, eos_tok=None):
    corpora = ['aochildes', 'open_subtitles', 'qed', 
               'switchboard', 'children_stories', 'bnc_spoken', 'simple_wikipedia',
               'wikipedia', 'cbt', 'gutenberg',]
    
    train_corpora = [ 'open_subtitles', #'aochildes', #'bnc_spoken',
               'children_stories', 'gutenberg_fixed', 'cbt', 
               'qed', 'simple_wikipedia', 'switchboard', 'wikipedia']
    
    train_corpora = [f'../babylm_data/babylm_10M/{corpus}.train' for corpus in train_corpora]
    train_corpora.append("/mnt/storage/nasimb/babylm_data/10M_log_rarity/bnc_spoken_log_rarity.train")
    train_corpora.append("/mnt/storage/nasimb/babylm_data/10M_log_rarity/aochildes_log_rarity.train")
    print(train_corpora)
    dev_corpora = [f'../babylm_data/babylm_dev/{corpus}.dev' for corpus in corpora]
    test_corpora = [f'../babylm_data/babylm_test/{corpus}.test' for corpus in corpora]
    return create_dataset_dict(train_corpora, dev_corpora, test_corpora, bos_tok, eos_tok)


    

