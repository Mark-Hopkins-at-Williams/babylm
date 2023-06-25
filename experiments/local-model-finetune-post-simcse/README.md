finetuning bert checkpoint trained with strict small and then with simcse in order to casr the model into AutoModelForMaskedLM rather than BertForCL.

- training for 1 epoch, on  all datasets in strict small
- eval every 1000 steps
- pretrained tokenizer 
- truncating longer sequences
- context length 128
