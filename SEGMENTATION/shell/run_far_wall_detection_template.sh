#!/bin/sh

#PBS -l walltime=15:00:00
#PBS -l nodes=1:ppn=12
#PBS -l mem=6GB
#PBS -m ae
#PBS -e outCluster/FW_U.err
#PBS -o outCluster/FW_U.out
#PBS -N infer_FW_U
#PBS -M laine@creatis.insa-lyon.fr

source ~/anaconda3/etc/profile.d/conda.sh
conda activate TF2.4

# --- path to the script
path=/home/laine/REPOSITORIES/carotid_US_DL_tool/consortium_meiburger/segmentation
cd $path

# --- run the the script
PYTHONPATH=$path python scripts/run_far_wall_detection.py -param set_parameters_NL_borders_union_F1.py
