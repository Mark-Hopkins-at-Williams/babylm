Notes:

- tokenizer is imported from huggingface
- experiment for sanity chack, results should be identical to the gpt2 experiment
- trains on all 10 datasets in strict small 

gpt2-concat-aochiles-14k:
- prev used file aochildes_modified_length_14k
- gutenberg-fixed

gpt2-concat-aochildes-16k:
- read every 10 sentences every 1000 lines
- look for meaningful and grammatically correct sentences or phrases 
- 16k -> first time 7 out of 10 were good
- sorted by length


Run the following on Appa:
    sbatch train_gpt2_concatenation_ss.sh