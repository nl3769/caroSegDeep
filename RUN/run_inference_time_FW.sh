 #!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate caroSegDeep

WD=/home/laine/cluster/REPOSITORIES/caroSegDeep/SEGMENTATION
cd $WD

PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f0 -overlap 96
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f0 -overlap 4
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f0 -overlap 8
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f0 -overlap 16
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f0 -overlap 32
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f0 -overlap 64


PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f1 -overlap 96
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f1 -overlap 4
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f1 -overlap 8
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f1 -overlap 16
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f1 -overlap 32
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f1 -overlap 64


PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f2 -overlap 96
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f2 -overlap 4
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f2 -overlap 8
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f2 -overlap 16
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f2 -overlap 32
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f2 -overlap 64


PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f3 -overlap 96
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f3 -overlap 4
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f3 -overlap 8
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f3 -overlap 16
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f3 -overlap 32
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f3 -overlap 64


PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f4 -overlap 96
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f4 -overlap 4
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f4 -overlap 8
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f4 -overlap 16
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f4 -overlap 32
PYTHONPATH=$WD python scripts/run_far_wall_detection_EXP.py -param set_parameters_inference_iccvg.py -fold f4 -overlap 64
