Notes:

- Tokenizer for this experiment is imported from huggingface
- trains the model with all 10 datasets in strict small
- uses DP for augmenting the tokenized sequences

Run the following on Appa:
    sbatch train_gpt2_dp_ss.sh
