from tokenizer import create_children_stories_dataset_dict, tokenize, TOKENIZER, CONTEXT_LENGTH
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


raw_datasets = create_children_stories_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)






TOKENIZER.pad_token = TOKENIZER.eos_token
data_collator = DataCollatorForLanguageModeling(TOKENIZER, mlm=False)
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")



config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(TOKENIZER),
    n_ctx=CONTEXT_LENGTH,
    bos_token_id=TOKENIZER.bos_token_id,
    eos_token_id=TOKENIZER.eos_token_id,
)
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
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
trainer.push_to_hub()

