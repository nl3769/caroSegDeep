#!/bin/bash


WD=~/Documents/REPO/caroSegDeep/SEGMENTATION
cd $WD
PYTHONPATH=$WD python scripts/run_IMC_segmentation.py -param set_parameters_inference_template.py
