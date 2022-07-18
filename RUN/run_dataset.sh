#!/bin/bash


WD=/home/laine/REPOSITORIES/caroSegDeep/SEGMENTATION
cd $WD
PYTHONPATH=$WD python run/run_dataset.py -param set_parameters_dataset_template.py
