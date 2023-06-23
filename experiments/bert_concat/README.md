Notes:

- Tokenizer trained from scratch
- Pre trained tokenizer cast into RobertaTokenizer due to bugs
- Uses concatenation for augmenting tokenized sequences
- Uses all 10 datasets in strict small
- finegrained training loop with eval steps every 500 steps and gradient accumalation set to 1
- context length 128
- mlm_probability: 0.2

second run notes:
- mlm_probability: 0.15
- sacve steps 1000, 20 epochs
- default tokenizer

thrid run:
- 512 contect length for purpose of compatibility with simcse


Run the following on Appa:
    sbatch train_bert_concat_ss.sh

