from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

def read_lines(filename):
    with open(filename) as reader:
        for line in reader:
            line = line.strip()
            if len(line) > 0:
                yield {'text': line}            

def create_dataset(filename):
    return Dataset.from_generator(lambda: read_lines(filename))

def create_dataset_dict(train_file, valid_file):
    result = DatasetDict()
    result['train'] = create_dataset(train_file)
    result['valid'] = create_dataset(valid_file)
    return result

def create_children_stories_dataset_dict():
    return create_dataset_dict('babylm_data/babylm_10M/children_stories.train',
                               'babylm_data/babylm_dev/children_stories.dev')



CONTEXT_LENGTH = 128
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def tokenize(element):
    outputs = TOKENIZER(element["text"], truncation=False)
    input_batch = []
    next_segment = []
    for input_ids in outputs["input_ids"]:
        next_segment.extend(input_ids)
        next_segment.append(TOKENIZER.eos_token_id)
        if len(next_segment) > CONTEXT_LENGTH:
            input_batch.append(next_segment[:CONTEXT_LENGTH])
            next_segment = next_segment[CONTEXT_LENGTH:]
    return {"input_ids": input_batch}

raw_datasets = create_children_stories_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)



