#!/bin/bash

WD=~/Documents/REPO/caroSegDeep/SEGMENTATION
cd $WD
PYTHONPATH=$WD python run/run_caro_seg_deep_train.py -param set_parameters_far_wall_training_template.py
