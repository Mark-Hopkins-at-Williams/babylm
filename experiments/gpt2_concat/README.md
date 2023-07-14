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

gpt2-concat-aochildes-length-15k:
- binary search for optimum cut for length sorted aochildes 

gpt2-concat-mod-datasets-rarity1-rarity-all-13k-2p6k:
- rerun of the "45e8878" experiment which includes 3 modified datasets aochildes-length-14k, cbt-rarity-2-.3, guten-2.2-1
- with post processing of all rarity clean up 
- 13k first place where 7 out of 10 are meaning ful phrases (not sentences)
- -2.6k first place inputs are longer than one word (721000)

gpt2-concat-mod-datatsets-rarity-all-iorder-e13k-e2.6k:
- rerun of the "de48df1" (above) experiment, however preserving the order of the sentences in each dataset
- sentences are ordered based on rarity, then a cut is estimated, list of ordered indicies of the sentences are cut, then the sentences are reordered based on the original order to preserve the context in shorter sentences
- the cutts are places in the same place as teh above experiment aproximately (total diff in the datasets less than 150 sents for comparison)

gpt2-concat-guten-mod-rarity-1k-p1k:
- modified teh gutenberg dataset to remove most refrences with deterministic text processing
- sorted teh resulting dataset based on gutenberg's internal token count
- at 1000 3 out of 10 sentences were meaningful and at -100 

gpt2-concat-guten-mod-rarity-e1k-ep1k:
- modified teh gutenberg dataset to remove most refrences with deterministic text processing
- sorted teh resulting dataset based on gutenberg's internal token count
- at 1000 3 out of 10 sentences were meaningful and at -100 
- after finding the cuts the dataset is reordered to represent the original order for comparison of the effectivitiy of this method

gpt2-concat-mod-datatsets-rarity-all-iorder-e13k:
- experiment to observe whether the worse performance is due to cutting the least frequent inputs

gpt2-concat-mod-datatsets-rarity-all-iorder-end-e2p6k:
- experiment to observe whether the worse performance is due to cutting the short but most frequent inputs

gpt2-concat-mod-datatsets-rarity-all-iorder-no-cut:
- rerun of 45e8878
- differences: order of datastes and the writing all datasets to a single file

gpt2-concat-guten-rarity-iroder-est-rarity-all-5k-2p5k:
- gutenberg sorted based on internal rarity cut where the best result so far (guten-rarity-all-5k-2.5k) is cut
- inputs reorderd after to replicate og datset minus the cut sentences 

gpt2-concat-cbt-rarity-iorder-2k-p3k:
- rerun of e486cf0 (cbt-rarity-2k-p3k), the cut from the best result (cbt-rarity-all-4.5k-.3k) no applicable to due to signifacnt index difference
- cbt-rarity-2k-p3k reordered to initial dataset order

gpt2-concat-aochildes-length-iorder-16k:
- rerun of 0709594 (aochildes-length-16k) but the dataset reordered

gpt2-concat-mod-datatsets-rarity-all-iorder-no-cut-repetition:
- rerun of 3dd6518 
- differences: repeated entries allowed to closer replicate 45e8878

gpt2-concat-guten-rarity-all-mod-repetition-iorder-5k-p5k:
- gutenberg sorted based on rarity of all tokens in strict small
- repeated entries not removed
- the refrences are removed with dtereministic text processing
- initial order of the dataset preserved

gpt2-concat-cbt-rarity-2k-p3k-rerun:
- rerun of e486cf0

gpt2-concat-mod-datasets-rarity1-rerun:
- rerun of 45e8878
- excatly the same code

gpt2-concat-aochildes-mod-no-repeating-sub-5p9k:
- removed inputs that have a repeating substring more than 3 times, 5.9k removed this way
- repeating lines in aochildes removed as well 80k -> 54k total
- gutenberg fixed

gpt2-concat-aochildes-mod-no-repeating-sub-5p9k-length-5k:
- same as the above experiment (fd6c646) but sorted with length 5k removed 
- at 5k 9 out of 10 are meaningful phrases
- initial order not preserved

gpt2-concat-guten-mod-rm-refrences-1p7k:
- fixed gutenberg to add space when merges happen (IMPORTANT) and slight modification to rm refrences which only works for arity sorted guten
- same deterministic text processing as "3c9c06e", effect isolated
- repetition removed
- tottal 2k sentences are the difference between this dataset and gutenberg_fixed

gpt2-concat-guten-mod-rm-ref-2k-rarity-2p5k-p13k:
- fixed gutenberg to add space when merges happen (IMPORTANT) and slight modification to rm refrences which only works for arity sorted guten -> 2k removed this way
- repetition removed
- original order not preserved
- intrnal rarity sorted -> read every 500 lines from least frequent at 2500 8 out of 10 score
- at -130 of most frequent 9 out of 10 meaningful phrases/sentences

gpt2-cocnat-aochildes-mod-no-repreating-sub-5p9k-length-15p5k:
- length sorted with length 15.5k removed, estimated based on best prev result (aochildes-len-16k) 
- initial order not preserved
- initial readline reset

gpt2-concat-cbt-mod-formatting-iorder:
- removed mid sentence captions, revoed distance between n't and 's
- unified "
- original order preserved

gpt2-concat-cbt-mod-formatting-rarity-all-4k:
- formatting as above
- original order not preserved
- 4k cur botha based on manual evaluation and cbt-rarity-all-4.5k-.8k

gpt2-concat-guten-mod-2k-rarity-all-4k-p12k:
- guten same text processing, sorted based on rarity all tokens
- cut placement both based on best result from guten-rarity-all-5k-2.5k and manual eval

gpt2-concat-simple-wiki-mod:
- deterministic text processing
- removed meaningless lines about style or web address
- removed the name before each definition
- removed cities aftyer line 60000

gpt2-cocnat-mod-datasets-txt-processing:
- aochildes_mod_no_repeating_sub.train
- cbt_mod_formatting_iorder.train
- guten_mod_rm_refrences_1p7k.train
- simple_wiki_mod.train

gpt2-dp-mod-datasets-txt-processing:
- aochildes_mod_no_repeating_sub.train
- cbt_mod_formatting_iorder.train
- guten_mod_rm_refrences_1p7k.train
- simple_wiki_mod.train

gpt2-concat-all-new-mod-datasets-rarity-all-iorder-13k-2p6k:
- rerun of c48e930 but few details in dataset generation fixed

gpt2-dp-all-mod-datasets-rarity-all-iorder-13k-2p6k:
- rerun of above with dp bfe2c1f

gpt2-cocnat-aochildes-mod-sub-length-10k:
- binary search for optimum cut after text processing

gpt2-concat-mod-datasets-txt-processing-rarity-all:
- aochildes_mod_no_repeating_sub_5p9k_length_15p5k.train
- cbt_mod_formatting_rarity_all_4k.train
- guten_mod_rarity_all_4k_p12k.train
- simple_wiki_mod.train

gpt2-dp-mod-datasets-txt-processing-rarity-all:
- dp version of the above (fb03238)

gp2-concat-guten-mod-rm-2p3k-rarity-all-5k-p22k:
- gutenberg removed refrences, raity all sorted cut around best results sofar

gpt2-concat-cbt-mod-formatting-iorder-rarity-all-4k:
- same as 0ce07fb but preserving the initial order

gpt2-dp-guten-rarity-all-5k-2.5k:
- repetition og the most successful cut (guten-rarity-all-5k-2.5k) for dp "d496465"

gpt2-concat-cbt-mod-formatting-rarity-all-no-cut:
- based on experiments whenever ioder was introduced regardless of teh cut, performance decreased
- test to see if rarity all order has any effect

gpt2-concat-all-mod-datasets1-rarity-all-iorder-c13k-c2p6k:
- rerun of "bfe2c1f" with currect end point cut, prev was cut at around 5k

gpt2-concat-all-mod-datasets1-rarity-all-iorder-c13k:
- rerun of above "8de31b6" with only the inital 13k cut

gpt2-concat-all-mod-datasets1-rarity-all-iorder-end-c2p6k:
- rerun of above "8de31b6" with only the end 2p6 cut

gpt2-concat-all-mod-datasets1-rarity-all-c13k-c2p6k-rev:
- rerun of above "8de31b6" with rarity all order (from eleast frequent to most) preserved

gpt2-cocnat-mod-datasets3-rarity-all:
- aochildes_length_16k.train
- cbt_mod_formatting_rarity_all_4k.train
- gutenberg_rarity_all_5k_2p5k.train

gpt2-concat-all-mod-datasets2-rarity-all-2k-13k:
- guten-rarity-all-5k-2.5k, aochildes-len-16k, cbt-rarity-all-4.5k-.8k 
- sorted all datasets based on rarity all, 2k from most frequent, 13k from least is cut

gpt2-cocnat-mod-datasets4-rarity-all-cbt-no-cut:
- aochildes_length_16k.train
- cbt_mod_formatting_rarity_all_no_cut.train
- gutenberg_rarity_all_5k_2p5k.train

gpt2-concat-mod-rm-2p3k-guten-rarity-all-no-cut:
- gutenberg sorted based on rarity of all datasets, no cut , modified to remove refrences

gpt2-concat-guten-rarity-all-no-cut:
- gutenberg sorted based on rarity of all datasets, no cut , no modification 

gpt2-concat-mod-datasets1-rarity-all-no-cut:
- experiment to see if just sorting all datasets based on rarity all the same as cl but with no sampling works
- mod datasets 1 -> aochildes 14k ...

gpt2-cocnat-mod-datasets1-rarity-all-5p5k-mostf:
- repetition of the b7e2c1f end cut (most frequent) preserving the rarity all sort order

gpt2-concat-mod-datasets1-iorder-rarity-all-5p5k:
- repetition of the b7e2c1f end cut (most frequent) reversed to initial order

gpt2-concat-guten-rarity-no-cut:
- baseline with gutenberg sorted based on internal token count and no cut

gpt2-concat-cbt-mod-formatting-rarity-all-no-cut-rev:
- 962efc8 reverse order

gpt2-concat-guten-mod-rm-rarity-all-no-cut-rev:
- 6e86132 reverse order

gpt2-concat-aochildes-mod-sub-rarity-all-no-cut-rev:
- aochildes modified (substring text processing) sorted based on all token counts no cutes in rev order

gpt2-concat-all-indv-rarity-all-no-cut:
- individually sorted based on all token counts

gpt2-concat-all-ind-txt-processing-rarity-all:
- same as above 46ad9cc just with the texy processed files

gpt2-concat-all-base-rarity-all-iorder-est-5p5k:
- repetition of gpt2-concat-mod-datasets1-iorder-rarity-all-5p5k 76682dc but using base datasets with no modification besides guten
- cut estimated based on cut in 76682dc

gpt2-concat-all-text-processign-rarity-all-iorder-est-5p5k:
- repetition of gpt2-concat-mod-datasets1-iorder-rarity-all-5p5k 76682dc but using base datasets modified with text processing
- cut estimated based on cut in 76682dc

gpt2-concat-cbt-mod-formatting-rarity-no-cut:
- repetition of b723054 with rarity sort instead of rarity all

gpt2-concat-cbt-rarity-no-cut:
- cbt no modification with rarity sort

gpt2-concat-cbt-rarity-all-no-cut:
- cbt rarity all sorted no modifications

gpt2-cocnat-guten-mod-rm-2k-rarity-no-cut:
- guten fixed passed tho mod sorted by rarity

gpt2-concat-aochildes-len-no-cut:
- aochildes length order no cut

gpt2-concat-aochildes-rarity-no-cut:
- aochildes rarity order no cut

gpt2-concat-aochildes-rarity-all-no-cut:
- aochildes rarity all order no cut

gpt2-concat-bnc-rarity-all-cut:

gpt2-concat-bnc-rarity-no-cut:

gpt2-concat-simple-wiki-rarity-no-cut:

gpt2-concat-simple-wiki-rarity-all-no-cut:

gpt2-concat-simple-wiki-mod-rarity-all-no-cut:
- modified to remove the few word title before each definition

gpt2-concat-all-base-rarity-all-iorder-8k:
- base datasets (no mod) cut 8k after rarity all sort
- initial order preserved

gpt2-concat-guten-rarity-all-end-2p5k:
- guten sorted by rarity all, only cutting the most frequent cut of the most successful result

gpt2-concat-cbt-rarity-all-end-p5k:
- cbt sorted by rarity all, only cutting the average of most frequent cut of the two most successful result

gpt2-concat-aochildes-rarity-end-3p3k:
- aochildes sorted based on rarity (best order) most frequent read every 500 lines at 52400 majority meaningful phrases non-repetitive

gpt2-concat-aochildes-mod-sub-1k-rarity-no-cut:
- aochildes mod after repetition removal only deletes 1k sentences, even tho the reported number is 5.9 before repetition removal

gpt2-concat-guten-mod-rarity-all-bnc-rarity:
- baseline plus:
bnc_rarity_no_cut
guten_mod_rm_2p3_rarity_all_no_cut

gpt2-concat-bnc-rarity-end-1p6:
- bnc soretd a=based on rarity 1600 cut from most frequent

gpt-concat-open-rarity-no-cut:
- open_subtitles sorted based on internal rarity, no cut

gpt2-concat-open-rarity-all-no-cut:
- open_subtitles sorted based on rarity all tokens, no cut

gpt2-concat-children-rarity-all-no-cut:
- children_stories sorted based on rarity all tokens, no cut

gpt2-concat-children-rarity-no-cut:
- children_stories sorted based on internal rarity, no cut

gpt2-concat-qed-rarity-no-cut:
- qed sorted based on internal rarity, no cut

Run the following on Appa:
    sbatch train_gpt2_concatenation_ss.sh