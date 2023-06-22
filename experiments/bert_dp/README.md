Notes:

- Tokenizer trained from scratch
- Pre trained tokenizer cast into RobertaTokenizer due to bugs
- Uses DP for augmenting tokenized sequences
- Uses all 10 datasets in strict small
- finegrained training loop with eval steps every 500 steps and gradient accumalation set to 1
- context length 128
- mlm_probability: 0.2

second run notes:
- mlm_probability: 0.15
- save steps 2000, 30 epochs
- default tokenizer


Run the following on Appa:
    sbatch train_bert_dp_ss.sh

