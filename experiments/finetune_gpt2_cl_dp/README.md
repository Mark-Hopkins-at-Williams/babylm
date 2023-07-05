Notes:

- finetuning the best model trained with cl on modifeid strict small (one layer rarity) 
- model imported /mnt/storage/nasimb/babylm/gpt2-cl-concat-rarity-mod-datasets-6
- Tokenizer for this experiment is imported from huggingface
- trains the model with all 10 datasets in strict small
- uses DP for augmenting the tokenized sequences


Run the following on Appa:
    sbatch /mnt/storage/nasimb/babylm/experiments/finetune_gpt2_cl_dp/finetune_gpt2_cl_dp_ss.sh

