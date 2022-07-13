'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import argparse
import importlib
import time
import wandb
from package_utils.compute_metrics      import compute_metrics_FW
from package_handler.sequence           import sequenceClassFW
import numpy                            as np
import matplotlib.pyplot                as plt

def get_patient_name(pfold, patient_list):

    with open(os.path.join(pfold, 'TestList.txt'), 'r') as f:
        lines = f.readlines()

    lines = [key.split('.')[0] + '.tiff' for key in lines]
    intersection_set = set.intersection(set(patient_list), set(lines))
    return list(intersection_set)
# ----------------------------------------------------------------------------------------------------------------------

def check_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
# ----------------------------------------------------------------------------------------------------------------------

def save_seg(p, seq, patient):
    ''' Save segmentation results in .txt format. '''
    path = os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'FAR_WALL_DETECTION')
    check_dir(path)
    FW_ = open(os.path.join(path, patient.split('.')[0] + ".txt"), "w+")
    for k in range(seq.annotationClass.borders_ROI['leftBorder'], seq.annotationClass.borders_ROI['rightBorder'] + 1, 1):
        FW_.write(str(k) + " " + str(seq.annotationClass.map_annotation[0, k, 0] / seq.scale) + "\n")
    FW_.close()
# ----------------------------------------------------------------------------------------------------------------------

def save_image(p, seq, patient):
    ''' Save image for visual inspection. '''
    img = np.zeros(seq.first_frame.shape + (3,))
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = seq.first_frame, seq.first_frame, seq.first_frame

    for k in range(seq.annotationClass.borders_ROI['leftBorder'], seq.annotationClass.borders_ROI['rightBorder'] + 1, 1):
        img[round(seq.annotationClass.map_annotation[0, k, 0] / seq.scale), k, 2]=0
        img[round(seq.annotationClass.map_annotation[0, k, 0] / seq.scale), k, 0] = 255
        img[round(seq.annotationClass.map_annotation[0, k, 0] / seq.scale), k, 1] = 0

    path = os.path.join(p.PATH_WALL_SEGMENTATION_RES, "IMAGES_FW")
    check_dir(path)
    plt.imsave(os.path.join(path, patient.split('.')[0] + ".png"), img.astype(np.uint8))
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # --- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser=argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of package_parameters required to execute the code.')
    my_parser.add_argument('--fold', '-fold', type=str)
    my_parser.add_argument('--overlap', '-overlap', type=int)

    arg=vars(my_parser.parse_args())
    param=importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])

    # --- we get package_parameters
    p=param.setParameters()

    # --- we get package_parameters
    p = param.setParameters()

    p.OVERLAPPING = arg['overlap']
    p.PATH_TO_FOLDS = os.path.join(p.PATH_TO_FOLDS, arg['fold'])


    p.PATH_TO_LOAD_TRAINED_MODEL_FW = os.path.join(p.PATH_TO_LOAD_TRAINED_MODEL_FW + arg['fold'].split('f')[-1], 'EXAMPLE')

    # --- get image name
    patient_name_list = os.listdir(p.PATH_TO_SEQUENCES)
    patient_name_list = get_patient_name(p.PATH_TO_FOLDS, patient_name_list)

    # --- create exec_time obj
    check_dir(p.PATH_WALL_SEGMENTATION_RES)
    if os.path.isfile(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "exec_time.txt")):
        os.remove(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "exec_time.txt"))
    if os.path.isfile(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "nb_patches.txt")):
        os.remove(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "nb_patches.txt"))
    exec_time = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "exec_time.txt"), "w")
    nb_patches = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "nb_patches.txt"), "w")

    # --- init weight and biases
    config = {"spatial_resolution": p.DESIRED_SPATIAL_RESOLUTION,
              "patch_height": p.PATCH_HEIGHT,
              "patch_width": p.PATCH_WIDTH,
              "overlapping": p.OVERLAPPING,
              "inference": "FW",
              "training": False,
              "fold": arg['fold']}

    wandb.init(project="caroSegDeep", entity="nl37", dir=p.PATH_WALL_SEGMENTATION_RES, config=config, name='FW_detection_shift_' + str(arg['overlap']) + '_' + arg['fold'])

    # --- launch process
    incr = 0
    for patient_name in patient_name_list:
        incr += 1
        print(f' ########### PROGRESSON {incr} / {len(patient_name_list)} ########### ')
        print("Current processed patient: ", patient_name)

        # --- path to the data
        path_seq = os.path.join(p.PATH_TO_SEQUENCES, patient_name)
        path_borders = os.path.join(p.PATH_TO_BORDERS, patient_name.split('.')[0] + "_borders.mat")

        # --- create the object sequenceClass
        seq = sequenceClassFW(sequence_path =path_seq, path_borders=path_borders, patient_name=patient_name, p=p)

        # --- launch the segmentation
        t = time.time()
        seq.launch_seg_far_wall(p=p)
        elapsed = time.time() - t

        # --- save execution timer and number of patches
        exec_time.write(str(elapsed) + "\n")
        nb_patches.write(str(len(seq.predictionClassFW.patches)) + "\n")

        # --- save segmentation results
        save_seg(p, seq, patient_name)

        max_val, mean_val = compute_metrics_FW(p.PATH_LO_LOAD_GT, patient_name, 'A1',
                                                         seq.annotationClass.map_annotation[0,] / seq.scale, seq.scale)

        wandb.log({"Patient": patient_name.split('.')[0],
                   "Max diff": max_val,
                   "Mean diff": mean_val,
                   "nb_patches": len(seq.predictionClassFW.patches),
                   "exec_time (full)": elapsed})

        # --- save image with far wall delineation
        save_image(p, seq, patient_name)


