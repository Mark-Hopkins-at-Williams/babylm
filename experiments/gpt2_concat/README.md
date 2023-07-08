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


- no self in rarity all does not work

gpt2-concat-cbt-rarity-all-5.75k-.55k:
- binary search for optimum cut for rarity sorted cbt

gpt2-concat-guten-rarity-all-7k-3k:
- search for optimum cut for rarity sorted gutenberg 

gpt2_concat_aochildes_len_16k_rarity_all_3k_.95k:
- binary search for optimum cut for rarity sorted aochildes after a 16k len based cut


gpt2-concat-aochildes-length-16k-rarity-all-no-self-4k-1.2k:
- rerun of the best result with aochildes
- remove aochildes-len-16k from the token count database

gp2-concat-top-for-aochildes-cbt-guten:
- a gpt2 conat model trained on the best results acuired form aochildes, cbt, and guten
- aochildes-len-16k-rarity-all-4k-1.2k
- cbt-rarity-all-7k-0.8k
- guten-rarity-all-5k-2.5k

gpt2-concat-bnc-rarity-12k-1p5k:
- 12k is first where 8 out 10 sentences or phrases are meaningful 
- most frequent: checked every 500, at 69000 all meaningful but short sentences

gpt2-concat-bnc-rarity-all-15k-1k:
- 15k is first where 8 out 10 sentences or phrases are meaningful 
- most frequent: checked every 500, at 69500 all meaningful but short sentences

gp2-concat-longer-top3-aochildes-cbt-guten:
- a gpt2 conat model trained on the best results acuired form aochildes, cbt, and guten
- aochildes-len-16k
- cbt-rarity-all-4.5k-0.3k
- guten-rarity-all-5k-2.5k

gpt2-concat-guten-rarity-all-3.5k-1.8k:
- binary search for optimum cut for rarity sorted guten 

gpt2-concat-guten-rarity-5k-2.5k:
- repetition of the most succesful cot for guten with a different sorting algorithm
- based on the token counts of the dataset itself 

gpt2-concat-all-rarity-all-29k-3k:
- realized we could sort all sentences based on rarity all, probably much more efficient, 
- then the only modification to each dataset is based on internal information i.e rarity based on self or length
- aochildes unmodidies, gurtenberg-fixed
- 29k -> 7 put of 10 were meaningful
- -3k (740k) first where inputs were longer than a single word
- read every 1000 lines

gpt2-concat-all-rarity-all-29k-3k:
- aochildes_length_16k, gurtenberg-fixed
- read every 1000 lines
- 30k -> first time where 7 put of 10 were meaningful (8 out of 10 in hits case but in 29k 6out of 10)
- -3k (724k) first where inputs were longer than a single word


Run the following on Appa:
    sbatch train_gpt2_concatenation_ss.sh