#!/bin/sh

#PBS -l walltime=15:00:00
#PBS -l nodes=1:ppn=12
#PBS -l mem=6GB
#PBS -m ae
#PBS -e outCluster/error_IMC_MEIB_F0.err
#PBS -o outCluster/output_IMC_MEIB_F0.out
#PBS -N IMC_SEG_MEIB_F0
#PBS -M laine@creatis.insa-lyon.fr

source ~/anaconda3/etc/profile.d/conda.sh
conda activate TF2.4

# --- path to the script
WD=/home/laine/REPOSITORIES/carotid_US_DL_tool/codes/segmentation
# --- run the the script
cd $WD
PYTHONPATH=$WD python scripts/run_segmentation.py -param set_parameters_NL_IMC_MEIB_01_F0.py
