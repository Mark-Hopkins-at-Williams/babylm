Notes:

- Tokenizer for this experiment is imported from huggingface
- trains the model with the 9 datasets in strict smal, all except aochild
- aochildes dataset is modified to start from line 20000
- uses concatenation for augmenting the tokenized sequences

Run the following on Appa:
    sbatch /home/nasimb/babylm/experiments/modified_aochild/train_gpt2_modified_aochild_ss.sh

