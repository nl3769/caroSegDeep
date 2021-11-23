#!/bin/sh

#PBS -l walltime=30:00:00
#PBS -l nodes=1:ppn=2
#PBS -l mem=6GB
#PBS -m ae
#PBS -e outCluster/error_IMC_CUBS_F3_PB.err
#PBS -o outCluster/output_IMC_CUBS_F3_PB.out
#PBS -N IMC_CUBS_SEG_F3
#PBS -M laine@creatis.insa-lyon.fr

source ~/anaconda3/etc/profile.d/conda.sh
conda activate TF2.4

# --- path to the script
path=/home/laine/REPOSITORIES/PHD/carotid_US_DL_tool/SEGMENTATION/SEGMENTATION
cd $path

# --- run the the script
PYTHONPATH=$path python scripts/run_segmentation.py -param set_parameters_NL_FW_CUBS_F3.py
