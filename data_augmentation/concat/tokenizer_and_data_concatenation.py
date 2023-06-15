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

def create_dataset_dict(train_file_names, valid_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    result['valid'] = create_dataset(valid_file_names)
    return result

def create_multiple_files_dataset_dict():
    return create_dataset_dict(["../babylm_data/babylm_10M/switchboard.train",
                                "../babylm_data/babylm_10M/qed.train", 
                                "../babylm_data/babylm_10M/open_subtitles.train", 
                                "../babylm_data/babylm_10M/wikipedia.train", 
                                "../babylm_data/babylm_10M/bnc_spoken.train", 
                                "../babylm_data/babylm_10M/cbt.train", 
                                "../babylm_data/babylm_10M/children_stories.train", 
                                "../babylm_data/babylm_10M/gutenberg.train", 
                                "../babylm_data/babylm_10M/simple_wikipedia.train"], 
                               ["../babylm_data/babylm_dev/switchboard.dev",
                                "../babylm_data/babylm_dev/qed.dev",
                                "../babylm_data/babylm_dev/open_subtitles.dev", 
                                "../babylm_data/babylm_dev/wikipedia.dev", 
                                "../babylm_data/babylm_dev/bnc_spoken.dev", 
                                "../babylm_data/babylm_dev/cbt.dev", 
                                "../babylm_data/babylm_dev/children_stories.dev", 
                                "../babylm_data/babylm_dev/gutenberg.dev", 
                                "../babylm_data/babylm_dev/simple_wikipedia.dev"])


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
    


raw_datasets = create_multiple_files_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
