Notes:

- tokenizer is imported from huggingface
- experiment for sanity chack, results should be identical to the gpt2 experiment
- trains on all 10 datasets in strict small 
- finegrained training loop with eval steps every 500 steps and gradient accumalation set to 1
- context length 128

Run the following on Appa:
    sbatch /home/nasimb/babylm/experiments/distilgpt2_concat/train_distilgpt2_concat_ss.sh