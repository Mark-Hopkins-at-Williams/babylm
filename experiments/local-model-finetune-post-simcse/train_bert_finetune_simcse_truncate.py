from tokenize_and_data_bert_finetune_simcse import create_multiple_files_dataset_dict, tokenize, TOKENIZER, CONTEXT_LENGTH 
from transformers import AutoTokenizer, RobertaConfig, RobertaForMaskedLM , BertConfig, BertForMaskedLM
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments

raw_datasets = create_multiple_files_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, #remove_columns=raw_datasets["train"].column_names,
    load_from_cache_file=False
)
tokenized_datasets["train"].set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_datasets["valid"].set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=TOKENIZER, mlm=True, mlm_probability=0.15
)
tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True,  collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32,  collate_fn=data_collator)


model = AutoModelForMaskedLM.from_pretrained("/home/nasimb/SimCSE/result/unsup-simcse-bert-concat-2")

eval_logging_ckp_steps = 2000

args = TrainingArguments(
    output_dir="bert-concat-2-finetune-simcse-truncate",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=eval_logging_ckp_steps,
    logging_steps=eval_logging_ckp_steps,
    gradient_accumulation_steps=1,
    num_train_epochs=30,
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

trainer.train(resume_from_checkpoint=False)
trainer.push_to_hub()


