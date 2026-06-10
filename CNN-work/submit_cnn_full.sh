#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --account=PAS2608
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=10
#SBATCH --mail-user=laurenbryant
#SBATCH --output=cnn_train_%j.log

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "----------------------------"

cd $HOME/CNN-work
source ~/venvs/cnn/bin/activate
python cnn_full.py

echo "---------------------------"
echo "Job ended at: $(date)"