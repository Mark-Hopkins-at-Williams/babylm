#!/bin/sh
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH --mem=100G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:2        # Request two GPUs

python /mnt/storage/nasimb/babylm/experiments/gpt2_dp_cl/train_gpt2_dp_cl.py
cd ../evaluation-pipeline
python babylm_eval.py /mnt/storage/nasimb/babylm/gpt2-dp-cl-rarity-7-138k decoder