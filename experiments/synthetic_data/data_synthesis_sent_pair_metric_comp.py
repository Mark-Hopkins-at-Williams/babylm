from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
import random
from transformers import BertTokenizerFast, RobertaTokenizer
from collections import Counter
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import util

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def read_lines(filenames):
    for filename in filenames:
        with open(filename) as reader:
            n = 0
            for line in reader:
                line = line.strip()
                if len(line) > 0:
                    yield {'text': line}

def create_dataset(filenames):
    return Dataset.from_generator(lambda: read_lines(filenames))

def create_dataset_dict(train_file_names):
    result = DatasetDict()
    result['train'] = create_dataset(train_file_names)
    return result

def create_multiple_files_dataset_dict(one_dataset):
    if one_dataset:
        corpora = ['wikipedia',]
    else:
        corpora = ['bnc_spoken', 'open_subtitles', 'aochildes', 
               'children_stories', 'cbt', 'gutenberg_fixed', 
               'qed', 'simple_wiki_mod', 'switchboard', 'wikipedia']
    print(corpora)
    train_corpora = [f'/mnt/storage/nasimb/babylm_data/babylm_10M/{corpus}.train' for corpus in corpora]
    return create_dataset_dict(train_corpora)
      

#TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
CONTEXT_LENGTH = 128

class Gpt2Parameters:
    model_arch = "gpt2"
    is_mlm = False
    explicit_bos_token = True
    explicit_eos_token = True
    pad_token = '[PAD]'
    context_length = CONTEXT_LENGTH
    
params = Gpt2Parameters()
TOKENIZER = AutoTokenizer.from_pretrained(params.model_arch)
model = AutoModelForCausalLM.from_pretrained("/mnt/storage/nasimb/babylm/all-base-guten-rarity-all-2p5k-rerun").to(torch_device)


def tokenize(element):
    outputs = TOKENIZER(element["text"], truncation=False)  
    return {"input_ids": outputs["input_ids"]}


Based_on_target_dataset = False

if not Based_on_target_dataset:

    raw_datasets = create_multiple_files_dataset_dict(Based_on_target_dataset)
    
    raw_datasets_one = create_multiple_files_dataset_dict(not Based_on_target_dataset)
    tokenized_datasets_one = raw_datasets_one.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
        #load_from_cache_file=False
    )
    
else:
    raw_datasets = create_multiple_files_dataset_dict(Based_on_target_dataset)


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names,
    #load_from_cache_file=False
)

#count number of tokens in the train dataset
list_tokenized_seqs = list(tokenized_datasets["train"]["input_ids"])
token_counts = Counter()
total_num_tokens = 0
for seq in list_tokenized_seqs:
    for token in seq:
        token_counts[token] += 1
        total_num_tokens += 1
        
        
if not Based_on_target_dataset:
    list_tokenized_seqs = list(tokenized_datasets_one["train"]["input_ids"])

#calculate the order of raw sentences based on the rarity of the tokens in each dataset      
dict_ind_token_rarity = {i:(sum([token_counts[token] for token in list_tokenized_seqs[i]]) 
                            / len(list_tokenized_seqs[i])) 
                         for i in range(len(list_tokenized_seqs))}
max_rarity = max(list(dict_ind_token_rarity.values()))


#normalizing the counts
normalized_token_counts = Counter()
for token_id, count in token_counts.items():
    normalized_token_counts[token_id] = count / total_num_tokens
        
#calculate the order of raw sentences based on the rarity of the tokens in each dataset      
dict_ind_token_log_rarity = {i: -1 * sum([math.log(normalized_token_counts[token]) for token in list_tokenized_seqs[i]]) 
                         for i in range(len(list_tokenized_seqs))}
max_log_rarity = max(list(dict_ind_token_log_rarity.values()))
print(max_log_rarity, max_rarity)

#calculate the order of raw sentences based on token length
tokenized_seq_lengths = [len(x) for x in tokenized_datasets["train"]["input_ids"]]
dict_ind_token_length = {i:tokenized_seq_lengths[i] for i in range(len(tokenized_seq_lengths))}


sorted_indecies = sorted(dict_ind_token_rarity, key=lambda k:(dict_ind_token_rarity[k]))

#reorder the raw datatset
if Based_on_target_dataset:
    list_train_dataset_raw = list(raw_datasets["train"]["text"])
else:
    list_train_dataset_raw = list(raw_datasets_one["train"]["text"])

sorted_list_train_dataset_raw = [list_train_dataset_raw[i] for i in sorted_indecies]

#remove repeating instances from the list preserving the order, and cut
sorted_list_train_dataset_raw = list(dict.fromkeys(sorted_list_train_dataset_raw))


total_sim_metrics_all_sents = [0, 0, 0, 0, 0, 0, 0, 0]

linei = 1
with open('/mnt/storage/nasimb/babylm/experiments/synthetic_data/wiki_sent_pair_metric_comp_norms.train', 'w') as f:
    while linei < 1000: #len(sorted_list_train_dataset_raw)//2:
        mostf_input = sorted_list_train_dataset_raw[-linei]
        start = mostf_input[:min(50, len(mostf_input)//2)].rsplit(" ", 1)[0]
        #make sure start has 2 word min
        while start.count(" ") < 2:
            linei += 1
            mostf_input = sorted_list_train_dataset_raw[-linei]
            start = mostf_input[:min(50, len(mostf_input)//2)].rsplit(" ", 1)[0]
        
        model_inputs = TOKENIZER(start, return_tensors='pt').to(torch_device)
        output = model.generate(
        **model_inputs,
        max_new_tokens=128,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        no_repeat_ngram_size=2,
        min_new_tokens = max(5, len(model_inputs['input_ids'])*2/3)
        )
        res=TOKENIZER.decode(output[0], skip_special_tokens=True)
        
        #metrics for the newly generated sent
        log_rarity_res = sum([math.log(normalized_token_counts[int(token)] )#if int(token) in normalized_token_counts else 1/total_num_tokens) 
                              for token in output[0][:-1]]) #ignore the end token at end
        rarity_res = sum([token_counts[int(token)] for token in output[0][:-1]]) / len(output[0][:-1])
        len_res = len(output[0][:-1])

        f.write(f"{linei}: {start}\n")
        f.write(f"{res}\n")
        
        def top_ten_most_similar_in_dataset(sorted_list_inputs):
            most_similar_inputs = [list_train_dataset_raw[i] for i in sorted_list_inputs[:40]] 
            most_similar_inputs = list(dict.fromkeys(most_similar_inputs))
            if len(most_similar_inputs) > 10:
                most_similar_inputs = most_similar_inputs[:10]
            return most_similar_inputs
        
        #log-rarity as similarity metric
        closest_ids_log_rarity = sorted(dict_ind_token_log_rarity, 
                                           key=lambda k:(abs(abs(dict_ind_token_log_rarity[k])-abs(log_rarity_res))))
        closest_inputs_log_rarity = top_ten_most_similar_in_dataset(closest_ids_log_rarity)
            
        #rarity
        closest_ids_rarity = sorted(dict_ind_token_log_rarity, 
                                    key=lambda k:abs(abs(dict_ind_token_rarity[k])-abs(rarity_res)))
        closest_inputs_rarity = top_ten_most_similar_in_dataset(closest_ids_rarity)
        
        #rarity + log rarity
        closest_ids_rarity_log_rarity = sorted(dict_ind_token_log_rarity, 
                                    key=lambda k:abs(abs(dict_ind_token_rarity[k])-abs(rarity_res))
                                    + (abs(abs(dict_ind_token_log_rarity[k])-abs(log_rarity_res))))
        closest_inputs_rarity_log_rarity = top_ten_most_similar_in_dataset(closest_ids_rarity_log_rarity)
        
        #len + rarity
        closest_ids_rarity_len = sorted(dict_ind_token_log_rarity, 
                                    key=lambda k:abs(abs(dict_ind_token_rarity[k])-abs(rarity_res))
                                    + abs(dict_ind_token_length[k] - len_res))
        closest_inputs_rarity_len = top_ten_most_similar_in_dataset(closest_ids_rarity_len)   
        
        #len + rarity + log_rarity 
        closest_ids_rarity_log_rarity_len = sorted(dict_ind_token_log_rarity, 
                                    key=lambda k:abs(abs(dict_ind_token_rarity[k])-abs(rarity_res))
                                    + abs(abs(dict_ind_token_log_rarity[k])-abs(log_rarity_res))
                                    + abs(dict_ind_token_length[k] - len_res))
        closest_inputs_rarity_log_rarit_len = top_ten_most_similar_in_dataset(closest_ids_rarity_log_rarity_len)
        
        #normalized rarity + log_rarity 
        closest_ids_normalized_rarity_log_rarity = sorted(dict_ind_token_log_rarity, 
                                    key=lambda k:abs(abs(dict_ind_token_rarity[k]/max_rarity)-abs(rarity_res/max_rarity))
                                    + abs(abs(dict_ind_token_log_rarity[k]/max_log_rarity)-abs(log_rarity_res/max_log_rarity)))
        closest_inputs_normalized_rarity_log_rarity = top_ten_most_similar_in_dataset(closest_ids_normalized_rarity_log_rarity)
        
        # mormalized log-rarity as similarity metric
        closest_ids_normalized_log_rarity = sorted(dict_ind_token_log_rarity, 
                                           key=lambda k:(abs(abs(dict_ind_token_log_rarity[k]/max_log_rarity)-abs(log_rarity_res/max_log_rarity))))
        closest_inputs_normalized_log_rarity = top_ten_most_similar_in_dataset(closest_ids_normalized_log_rarity)
            
        #normalized rarity
        closest_ids_normalized_rarity = sorted(dict_ind_token_log_rarity, 
                                    key=lambda k:abs(abs(dict_ind_token_rarity[k]/max_rarity)-abs(rarity_res/max_rarity)))
        closest_inputs_normalized_rarity = top_ten_most_similar_in_dataset(closest_ids_normalized_rarity)
           
           
        res_tokenized = TOKENIZER(res, add_special_tokens=False, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            logits = model(**res_tokenized).logits[0]     

        def compare_top_ten_metric(top_ten_inputs, metric_name, total_sim_metric_all_sents):
            total_sim_this_sent = 0
            f.write(f"***{metric_name}***\n")
            for j in range(len(top_ten_inputs)):
                sent = top_ten_inputs[j]
                tokenized_sent = TOKENIZER(sent, add_special_tokens=False, return_tensors='pt').to(torch_device)
                with torch.no_grad():
                    logits_sent = model(**tokenized_sent).logits[0]   
                cos_sim = util.pytorch_cos_sim(logits.mean(dim=0), logits_sent.mean(dim=0))
                total_sim_this_sent += float(cos_sim)
                f.write(str(j) +  ")  "+ sent + "  sim: " + str(cos_sim) + "\n")
            f.write(f"total sum metric {metric_name} for this sent: {total_sim_this_sent}\n")
            f.write(f"\n")
            return float(total_sim_metric_all_sents) + float(total_sim_this_sent)
        
        metric_names = ["log-rarity", "rarity", "rarity + log-rarity",
                        "len + rarity", "len + rarity + log-rarity", 
                        "normalized(rarity + log-rarity)", "normalized log-rarity", 
                        "normalized rarity"]
        top_tens = [closest_inputs_log_rarity, closest_inputs_rarity,
                    closest_inputs_rarity_log_rarity, closest_inputs_rarity_len,
                    closest_inputs_rarity_log_rarit_len, closest_inputs_normalized_rarity_log_rarity,
                    closest_inputs_normalized_log_rarity, closest_inputs_normalized_rarity]
        for k in range(8):
            total_sim_metrics_all_sents[k] = compare_top_ten_metric(top_tens[k], 
                                                                    metric_names[k], 
                                                                    total_sim_metrics_all_sents[k])
                   
        final_sims = [metric_names[i] + ": " + str(total_sim_metrics_all_sents[i]) for i in range(8)] 
               
        for i in range(8):
            f.write(final_sims[i] + "\n")

        linei += 1
    

   

        
    


