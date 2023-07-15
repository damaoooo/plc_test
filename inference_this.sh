#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -J output
#SBATCH --partition=batch
#SBATCH -o output.%J.out
#SBATCH -e output.%J.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=200G
#run the application:

conda init bash
source /home/zhoul0e/.bashrc
conda activate ml
cd /ibex/user/zhoul0e/plc_test
python inference.py
