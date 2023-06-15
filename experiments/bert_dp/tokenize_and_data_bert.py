import sys
sys.path.append("/home/nasim/working_dir/Tokenizers")
from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
from DP_merging_buffer import dp_merge_inputs
import random
from transformers import BertTokenizerFast, RobertaTokenizer
from tokenizers import ByteLevelBPETokenizer

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
    return create_dataset_dict(["/home/nasim/babylm_data/babylm_10M/switchboard.train",
                                "/home/nasim/babylm_data/babylm_10M/qed.train", 
                                "/home/nasim/babylm_data/babylm_10M/open_subtitles.train", 
                                "/home/nasim/babylm_data/babylm_10M/wikipedia.train", 
                                "/home/nasim/babylm_data/babylm_10M/bnc_spoken.train", 
                                "/home/nasim/babylm_data/babylm_10M/cbt.train", 
                                "/home/nasim/babylm_data/babylm_10M/children_stories.train", 
                                "/home/nasim/babylm_data/babylm_10M/gutenberg.train", 
                                "/home/nasim/babylm_data/babylm_10M/simple_wikipedia.train"], 
                               ["/home/nasim/babylm_data/babylm_dev/switchboard.dev",
                                "/home/nasim/babylm_data/babylm_dev/qed.dev",
                                "/home/nasim/babylm_data/babylm_dev/open_subtitles.dev", 
                                "/home/nasim/babylm_data/babylm_dev/wikipedia.dev", 
                                "/home/nasim/babylm_data/babylm_dev/bnc_spoken.dev", 
                                "/home/nasim/babylm_data/babylm_dev/cbt.dev", 
                                "/home/nasim/babylm_data/babylm_dev/children_stories.dev", 
                                "/home/nasim/babylm_data/babylm_dev/gutenberg.dev", 
                                "/home/nasim/babylm_data/babylm_10M/simple_wikipedia.train"])
    """return create_dataset_dict(["../babylm_data/babylm_10M/switchboard.train",
                                "../babylm_data/babylm_10M/qed.train", 
                                "../babylm_data/babylm_10M/open_subtitles.train", 
                                "../babylm_data/babylm_10M/wikipedia.train"], 
                               ["../babylm_data/babylm_dev/switchboard.dev",
                                "../babylm_data/babylm_dev/qed.dev",
                                "../babylm_data/babylm_dev/open_subtitles.dev", 
                                "../babylm_data/babylm_dev/wikipedia.dev"])"""
    

CONTEXT_LENGTH = 512
TOKENIZER = RobertaTokenizer.from_pretrained("rbertT")


def tokenize(element):
    outputs = TOKENIZER(element["text"], truncation=False)   
    merged_inputs = dp_merge_inputs(outputs["input_ids"], CONTEXT_LENGTH, TOKENIZER.eos_token_id)
    return {"input_ids": merged_inputs}


"""raw_datasets = create_multiple_files_dataset_dict()
print(raw_datasets)
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
print(raw_datasets)
"""

