Notes:

- Tokenizer trained from scratch
- Pre trained tokenizer cast into RobertaTokenizer
- Uses DP for augmenting tokenized sequences
- Uses all 10 datasets in strict small
- context length 128
- training loop eval steps 1000, 30 epochs

Run the following on Appa:
    sbatch /home/nasimb/babylm/experiments/roberta_dp/train_roberta_dp_ss.sh
