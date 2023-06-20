Notes:

- buggy
- Tokenizer trained from scratch, vocab-size 30k
- Pre trained tokenizer cast into RobertaTokenizer
- Uses concatenation for augmenting tokenized sequences
- Uses all 10 datasets in strict small
- context length 128
- training loop eval steps 1000, 30 epochs
- mlm_probability: 0.15

Run the following on Appa:
    sbatch /home/nasimb/babylm/experiments/roberta_concat/train_roberta_concat_ss.sh
