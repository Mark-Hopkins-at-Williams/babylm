Notes:

- Tokenizer for this experiment is imported from huggingface
- trains the model with all 10 datasets in strict small
- uses DP for augmenting the tokenized sequences
- finegrained training loop with eval steps every 500 steps and gradient accumalation set to 1
- context length 128

Run the following on Appa:
    sbatch /home/nasimb/babylm/experiments/distilgpt2_dp/train_distilgpt2_dp_ss.sh

