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

gpt2-concat-aochildes-16k+6k:
- read every 10 sentences every 1000 lines
- look for meaningful and grammatically correct sentences or phrases 
- 22k -> first time 10 out of 10 were good
- sorted by length

gpt2-concat-aochildes-length-16k-rarity_all-4k-1.2k:
- read every 10 sentences every 1000 lines
- look for meaningful and grammatically correct sentences or phrases 
- 4k -> first time 8 out of 10 were good
- from the end of the dataset read 10 inputs every 500 lines and 38500 firstime 8 out of 10 were good
- sorted by rarity of tokens among all tokens of strict small


gpt2-concat-aochildes-length-16plus6k-rarity-all-3k-p6k
- read every 10 sentences every 1000 lines
- look for meaningful and grammatically correct sentences or phrases 
- 3k -> first time 8 out of 10 were good
- from the end of the dataset read 10 inputs every 500 lines and 33000 firstime 8 out of 10 were good
- aochildes-length-16plus6k sorted by rarity of tokens among all tokens of strict small

gpt2-concat-cbt-rarity-all-7k-0.8k:
- good sent: less than 3 odd words (ie ye instead of you), meaningful
- 7k first time 7 out of 10 sentences resemble modern english
- from end, 24500 first time all 10 are meaning full, checked every 500, starting from 25000

Run the following on Appa:
    sbatch train_gpt2_concatenation_ss.sh