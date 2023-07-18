from tokenizer_and_data_gpt2_concatenation import create_multiple_files_dataset_dict
import os
import sys
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
os.environ["TOKENIZERS_PARALLELISM"] = "false" # suppresses a transformers warning


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


def train(model_dir):
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
            while length + len(next_segment) <= params.context_length:
                next_segment.extend(input_ids)
            else:
                input_batch.append(next_segment)
                next_segment = []
                next_segment.extend(input_ids)
        return {'input_ids': input_batch}

    raw_datasets = create_multiple_files_dataset_dict()
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

    eval_logging_ckp_steps = 500
    
    args = TrainingArguments(
        output_dir=model_dir,
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
        tokenizer=tokenizer,
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

if __name__ == "__main__":
    train_dir = sys.argv[1]
    train(train_dir)
