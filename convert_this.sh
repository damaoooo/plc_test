#!/bin/bash
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32
#SBATCH -J output
#SBATCH -o output.%J.out
#SBATCH -e output.%J.err
#SBATCH --time=6-23:00:00
#SBATCH --mem=300G
#run the application:


source /home/zhoul0e/.bashrc
conda init bash
source /ibex/user/zhoul0e/.bashrc
conda activate ml
cd /ibex/user/zhoul0e/plc_test
module load singularity/3.6

# singularity exec -B /ibex/tmp/zhoul0e -B /ibex/user/zhoul0e /ibex/user/zhoul0e/cpg_v0.1.sif python3 /ibex/user/zhoul0e/plc_test/convert_to_json.py
/ibex/user/zhoul0e/miniconda3/envs/ml/bin/python /ibex/user/zhoul0e/plc_test/convert_to_json.py