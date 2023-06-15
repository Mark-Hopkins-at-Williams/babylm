Notes:

- Tokenizer trained from scratch
- Pre trained tokenizer cast into RobertaTokenizer due to bugs
- Uses DP for augmenting tokenized sequences
- Uses all 10 datasets in strict small

Run the following on Appa:
    sbatch train_bert_dp_ss.sh

