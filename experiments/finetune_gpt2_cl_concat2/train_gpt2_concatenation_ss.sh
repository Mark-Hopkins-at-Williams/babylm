#!/bin/sh
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-16:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH --mem=100G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:2        # Request two GPUs

python /mnt/storage/nasimb/babylm/experiments/finetune_gpt2_cl_concat2/finetune_gpt2_cl_concat.py
cd ../evaluation-pipeline
python babylm_eval.py /mnt/storage/nasimb/babylm/cl-rarity-all-base-iorder-5p5k-finetune-guten-rarity-all-2p5k decoder