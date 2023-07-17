#!/bin/sh
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-20:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH --mem=100G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1        # Request one GPUs

cd ../evaluation-pipeline
./finetune_all_tasks.sh '/mnt/storage/nasimb/babylm/gpt2-concat-all-base-rarity-all-iorder-est-5p5k'