from tokenizer_and_data_gpt2_concatenation import create_multiple_files_dataset_dict, tokenize, TOKENIZER, CONTEXT_LENGTH 
from transformers import GPT2LMHeadModel, AutoConfig
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


raw_datasets = create_multiple_files_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
    load_from_cache_file=False
)

TOKENIZER.pad_token = TOKENIZER.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm=False)

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32,  collate_fn=data_collator, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32,  collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=32,  collate_fn=data_collator)


config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(TOKENIZER),
    n_ctx=CONTEXT_LENGTH,
    bos_token_id=TOKENIZER.bos_token_id,
    eos_token_id=TOKENIZER.eos_token_id,
)
model = GPT2LMHeadModel(config)

eval_logging_ckp_steps = 500

args = TrainingArguments(
    output_dir="cbt-guten-log-rarity-all-no-cut-mixed",
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

