#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate creatis

working_directory=~/Documents/REPO/caroSegDeep/SEGMENTATION
cd $working_directory
PYTHONPATH=$working_directory python scripts/run_IMC_segmentation.py -param set_parameters_inference_template.py
