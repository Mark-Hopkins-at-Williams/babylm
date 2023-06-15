import sys
sys.path.insert(0, '/home/nasim/babylm/Tokenizers')
from tokenizer_dp_merging_buffer import create_multiple_files_dataset_dict, tokenize, TOKENIZER, CONTEXT_LENGTH
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_scheduler



raw_datasets = create_multiple_files_dataset_dict()
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
    load_from_cache_file=False
)

TOKENIZER.pad_token = TOKENIZER.eos_token
data_collator = DataCollatorForLanguageModeling(TOKENIZER, mlm=False)

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True,  collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32,  collate_fn=data_collator)



config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(TOKENIZER),
    n_ctx=CONTEXT_LENGTH,
    bos_token_id=TOKENIZER.bos_token_id,
    eos_token_id=TOKENIZER.eos_token_id,
)

model = GPT2LMHeadModel(config)


def crossEntropy_loss(inputs, logits):
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean()
  
    return loss_per_sample


def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


accelerator = Accelerator()

optimizer = AdamW(model.parameters(), lr=3e-5)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = 8
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)

gradient_accumulation_steps = 1
eval_steps = 1000


model.train()
completed_steps = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps, initial=completed_steps
    ):
        logits = model(batch["input_ids"]).logits
        loss = crossEntropy_loss(batch["input_ids"], logits)
        if step % 100 == 0:
            accelerator.print(
                {
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained("8e_9ds_dp_merging_buffer_model", save_function=accelerator.save)
            if accelerator.is_main_process:
                TOKENIZER.save_pretrained("8e_9ds_dp_merging_buffer_model")
