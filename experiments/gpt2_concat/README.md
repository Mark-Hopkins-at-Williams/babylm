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

gpt2-concat-len-16k-punc-dot:
- there's a lot of mismatch between the meaning of the sent and punctuation mark in aochildes
- experiment to see if fixing that helps

gpt2-concat-aochildes-len-16plus3k:
- binary search for optimum cut for length sorted aochildes

gpt2-concat-aochildes-len-17p5k:
- binary search for optimum cut for length sorted aochildes

gpt2_concat_aochildes_len_16k_rarity_all_6k_1.2k:
- search for optimum cut for rarity sorted aochildes after a 16k len based cut

gpt2_concat_aochildes_len_16k_rarity_all_2k_p7k:
- search for optimum cut for rarity sorted aochildes after a 16k len based cut

gpt2-concat-cbt-rarity-all-4.5k-0.3k:
- search for optimum cut for rarity sorted cbt 

gpt2-concat-cbt-rarity-all-no-cbt-7k-0.8k:
- sort the inputs in cbtbased on token counts in all datasets besides cbt 
- might be helpful as cbt contains a lof of rare words thatare not ptresent in otherdatasetsand we want to remove iunputs containing high number of rare words
- cut inds are the same as prev best result

gpt2-concat-cbt-rarity-all-12k-0.8k:
- search for optimum cut for rarity sorted cbt 

gpt2-concat-guten-rarity-all-5k-2.5k:
- at 5k all sentences are meaningfule or are longer sentences with meaningful portions
- for the most frequent: checked every 500 lines, at 19000 7 out of 10 were complete sentences

gpt2-concat-guten-rarity-no-self-5k-2.5k:
- gutenberg is not included in token counts
- the cut placements roughly matches that of rarity-all in terms of numebr of meaningful sentences
- aligned for purpose of comparison


Run the following on Appa:
    sbatch train_gpt2_concatenation_ss.sh