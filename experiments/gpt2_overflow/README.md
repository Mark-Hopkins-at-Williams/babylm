Notes:

- tokenizer is imported from huggingface
- does not augment the tokenized sequences together at all. Each sentence is tokenized seperately. If ht elength of the tokes of a sentence is less than the context length then the sequence is padded enough for its length to match the context length
- trains on all 10 datasets in strict small 

Run the following on Appa:
    sbatch train_gpt2_overflow_ss.sh