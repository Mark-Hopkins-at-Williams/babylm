Notes:

- Tokenizer for this experiment is imported from huggingface
- trains the model with 9 datasets in strict small, all but aochild
- aochildes dataset modified to start from line 20000
- uses DP for augmenting the tokenized sequences

Second run:
- aochiles modified to contain only sentences longer than 10 chars.

Run the following on Appa:
    sbatch /home/nasimb/babylm/experiments/gpt2_dp_modified_aochild/train_gpt2_dp_mod_aochild_ss.sh

