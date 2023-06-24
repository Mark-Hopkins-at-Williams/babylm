finetuning the checkpoint published by SimCSE paper for pretrained bert base uncased, in order to casr the model into AutoModelForMaskedLM rather than BertForCL.

- training for 1 epoch, on  all datasets in strict small
- eval every 1000 steps
- pretrained tokenizer 
- context length 128
