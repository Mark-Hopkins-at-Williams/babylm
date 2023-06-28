Notes:

- Tokenizer for this experiment is imported from huggingface
- trains the model with all 10 datasets in strict small
- uses DP for augmenting the tokenized sequences
- uses curriculum training to order the sentences based on length or rarity before tokenization 


Run the following on Appa:
    sbatch /mnt/storage/nasimb/babylm/experiments/gpt2_dp_cl/train_gpt2_dp_cl_ss.sh

