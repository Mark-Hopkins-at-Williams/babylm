from tokenizer_and_data_gpt2_concatenation import create_multiple_files_dataset_dict
from transformers import GPT2LMHeadModel, AutoConfig
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoTokenizer

CONTEXT_LENGTH = 128

class Gpt2Parameters:
    model_arch = "gpt2"
    is_mlm = False
    explicit_bos_token = True
    explicit_eos_token = True
    pad_token = '[PAD]'
    context_length = CONTEXT_LENGTH

    def init_model(self, config):
        return GPT2LMHeadModel(config)
    
params = Gpt2Parameters()
TOKENIZER = AutoTokenizer.from_pretrained(params.model_arch)
TOKENIZER.padding_side = "right"
TOKENIZER.pad_token = TOKENIZER.eos_token

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
    
    """attention_batch = []
    next_segment = []
    for attention_mask in outputs["attention_mask"]:
        next_segment.extend(attention_mask)
        next_segment.append(0)
        while len(next_segment) >= CONTEXT_LENGTH:
            attention_batch.append(next_segment[:CONTEXT_LENGTH])
            next_segment = next_segment[CONTEXT_LENGTH:]"""
    return {"input_ids": input_batch}#, "attention_mask": attention_batch}

raw_datasets = create_multiple_files_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
    load_from_cache_file=False
)

if params.pad_token is not None:
        TOKENIZER.add_special_tokens({'pad_token': params.pad_token})
        
data_collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm=False)

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32,  collate_fn=data_collator, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32,  collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=32,  collate_fn=data_collator)

config = AutoConfig.from_pretrained(
        params.model_arch,
        vocab_size=len(TOKENIZER),
        n_ctx=params.context_length,
        bos_token_id=TOKENIZER.bos_token_id,
        eos_token_id=TOKENIZER.eos_token_id,
    )
model = params.init_model(config)

model.resize_token_embeddings(len(TOKENIZER)) 
model.config.pad_token_id = model.config.eos_token_id

eval_logging_ckp_steps = 500

args = TrainingArguments(
    output_dir="guten-len", 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=eval_logging_ckp_steps,
    logging_steps=eval_logging_ckp_steps,
    gradient_accumulation_steps=1,
    num_train_epochs=6,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=eval_logging_ckp_steps,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=TOKENIZER,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()
print("test set evaluation")
print("*******************************************")
print(trainer.evaluate(eval_dataset=tokenized_datasets["test"]))
print("*******************************************")
trainer.push_to_hub()

