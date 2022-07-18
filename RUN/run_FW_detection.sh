#!/bin/bash


WD=/home/laine/cluster/REPOSITORIES/caroSegDeep/SEGMENTATION/
cd $WD
PYTHONPATH=$WD python run/run_far_wall_detection.py -param set_parameters_inference_template.py
