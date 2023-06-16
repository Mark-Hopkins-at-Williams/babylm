from babydata_all_but_one_datasets import strict_small_leave_one_out
import os
import sys
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
os.environ["TOKENIZERS_PARALLELISM"] = "false" # suppresses a transformers warning


class Gpt2Parameters:
    model_arch = "gpt2"
    is_mlm = False
    explicit_bos_token = True
    explicit_eos_token = True
    pad_token = '[PAD]'
    context_length = 64

    def init_model(self, config):
        return GPT2LMHeadModel(config)


def train(model_dir, leave_out_dataset):
    params = Gpt2Parameters()
    tokenizer = AutoTokenizer.from_pretrained(params.model_arch)

    def tokenize(element):
        outputs = tokenizer(
            element["text"], 
            truncation=True,
            max_length=params.context_length,
            return_overflowing_tokens=True, # if a document goes over the context length, split it into multiple segments
            return_length=True,
        )
        input_batch = []
        next_segment = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length + len(next_segment) <= params.context_length:
                next_segment.extend(input_ids)
            else:
                input_batch.append(next_segment)
                next_segment = []
                next_segment.extend(input_ids)
        return {'input_ids': input_batch}

    raw_datasets = strict_small_leave_one_out(leave_out_dataset, None, tokenizer.eos_token)
    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    if params.pad_token is not None:
        tokenizer.add_special_tokens({'pad_token': params.pad_token})

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=params.is_mlm)

    config = AutoConfig.from_pretrained(
        params.model_arch,
        vocab_size=len(tokenizer),
        n_ctx=params.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = params.init_model(config)

    args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=1000,
        gradient_accumulation_steps=8,
        num_train_epochs=9,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=1000,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.push_to_hub()

if __name__ == "__main__":
    
    train_dir = sys.argv[1]
    
    corpora = ['aochildes', 'bnc_spoken', 'open_subtitles',
               'children_stories', 'cbt', 'gutenberg', 
               'qed', 'simple_wikipedia', 'switchboard', 'wikipedia']
    for dataset in corpora:
        leave_out_dataset = dataset
        
        train_dir += "_left_out_" + leave_out_dataset
        train(train_dir, leave_out_dataset)
