#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -J train_this
#SBATCH -o output.%J.out
#SBATCH -e output.%J.err
#SBATCH --time=3-12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=200G
#run the application:

conda init bash
source /ibex/user/zhoul0e/.bashrc
conda init bash
conda activate py310
cd /ibex/user/zhoul0e/plc_test
/ibex/user/zhoul0e/miniconda3/envs/py310/bin/python main.py
