#!/bin/bash

#PBS -l walltime=3:00:00
#PBS -l nodes=1:ppn=2
#PBS -l mem=12GB
#PBS -m ae
#PBS -e repo_test.err
#PBS -o repo_test.out
#PBS -N repo_test
#PBS -M laine@creatis.insa-lyon.fr


source ~/anaconda3/etc/profile.d/conda.sh
conda activate caroSegDeep

working_directory=~/Desktop/caroSegDeep/SEGMENTATION
cd $working_directory
PYTHONPATH=$working_directory python scripts/run_dataset.py -param set_parameters_dataset_template.py
