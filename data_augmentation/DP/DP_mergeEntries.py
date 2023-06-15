import random

    
def dp_min_cost_merge_intervals(input_lens, context_length):
    for start_ind in range(len(input_lens)):
        end_ind = start_ind
        plus_one_range = False
        len_remainder_ptl_merge = input_lens[end_ind] 
        if input_lens[end_ind] < context_length:
            len_remainder_ptl_merge += 1 
        
        if start_ind == 0:
            cost_ptl_merge = 1
            mem = {}
        else:
            cost_ptl_merge, mem = prev_opt_interval_endi(start_ind, mem, first_end)

        while len_remainder_ptl_merge <= context_length:
            plus_one_range = True
            end_ind += 1
            if end_ind >= len(input_lens):
                break
            len_remainder_ptl_merge += input_lens[end_ind] 
            if input_lens[end_ind] < context_length:
                len_remainder_ptl_merge += 1

        if plus_one_range: 
            end_ind -= 1
        if end_ind not in mem or mem[end_ind][1] > cost_ptl_merge:
            mem[end_ind] = (start_ind, cost_ptl_merge)
        
        if start_ind == 0:
            first_end = end_ind
    
    return mem


#returns the optimum cost (number of partitions) of the intervals that lead to this start ind
#If the prev index is not observed as an end_ind for index starti, choose the biggest end index (BE_ind) before starti
#but add 2 to cost, can prove with contradiction that this would be the optimum partition number
#in order for a new merge interval to start at the given index: we already looked at BE_ind + 1 as an start
#which means the end index of interval starting with BE_ind + 1 is after starti, and thus there is no interval path
#that starts at starti with lower cost than the one starting at BE_ind + 1, and ending at starti -1 
def prev_opt_interval_endi(start_ind, mem, first_end):
    if start_ind <= first_end:
        mem[start_ind-1] = (0, 1)
        return 1, mem
    
    if not (start_ind - 1) in mem:    
        max_endi_prev_start = 0
        for e in mem:
            if e > max_endi_prev_start and e < start_ind:
                max_endi_prev_start = e
        mem[start_ind-1] = (max_endi_prev_start+1, mem[max_endi_prev_start][1]+1)
    
    cost_ptl_merge = mem[start_ind-1][1] + 1
    
    return cost_ptl_merge , mem


#merges inputs based on optimum intervals calculated with dynamic programming
def merge_inputs(inputs, mem, end_token, input_lens, context_length):
    merged_inputs = []
    i = len(inputs) - 1
    while i > -1:
        end_ind = i
        start_ind = mem[i][0]
    
        merged = []
        for j in range(start_ind, end_ind+1):
            merged.extend(inputs[j])
        if input_lens[j] < context_length:
            merged.append(end_token)
        
        merged_inputs.append(merged) 
        i = start_ind - 1 
    return merged_inputs
    
    

def generate_test_data(seed, num_samples, min_length_sample, max_length_sample):
    inputs = []
    random.seed(seed)
    for i in range(num_samples):
        #generate random length of ones
        length = random.randint(min_length_sample, max_length_sample)
        inputs.append([1 for i in range(length)])
    
    return inputs,


#calculates optimal index intervals to merge inputs with minimum number of resulting merged-inputs 
#inputs are merged to generate longer sequences with length strictly less that the context_length
#and returns the list of merged inputs
def dp_merge_inputs(inputs, context_length, end_token):
    if not inputs:
        return
    
    input_lens = []
    for i in inputs:
        input_lens.append(len(i))
                
    mem_intervals = dp_min_cost_merge_intervals(input_lens, context_length)

    merged_inputs = merge_inputs(inputs, mem_intervals, end_token, input_lens, context_length)
    
    return merged_inputs




if __name__ == "__main__":
    inputs = generate_test_data(0, 100, 1, 200)
    merged_inputs = dp_merge_inputs(inputs, 100, "end")
