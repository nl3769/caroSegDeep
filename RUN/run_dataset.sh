#!/bin/bash

#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=90GB
#PBS -m ae
#PBS -e dataset_gen.err
#PBS -o dataset_gen.out
#PBS -N dataset_gen
#PBS -M laine@creatis.insa-lyon.fr


source ~/anaconda3/etc/profile.d/conda.sh
conda activate caroSegDeep

working_directory=/home/laine/REPOSITORIES/caroSegDeep/SEGMENTATION
cd $working_directory
PYTHONPATH=$working_directory python scripts/run_dataset.py -param set_parameters_dataset_template.py
