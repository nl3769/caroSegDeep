"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import importlib
import time
from functions.compute_metrics import compute_metrics
import wandb

from classes.sequence import sequenceClassIMC

def check_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
# ----------------------------------------------------------------------------------------------------------------------

def save_seg(p, seq, patient):
    """ Save segmentation results in .txt format. """

    path = os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'IMC_RES')
    check_dir(path)

    LI_ = open(os.path.join(path, patient.split('.')[0] + "-LI.txt"), "w+")
    MA_ = open(os.path.join(path, patient.split('.')[0] + "-MA.txt"), "w+")
    for k in range(seq.annotationClass.borders['leftBorder'], seq.annotationClass.borders['rightBorder'], 1):
        LI_.write(str(k) + " " + str(seq.annotationClass.map_annotation[1, k, 0] / seq.scale) + "\n")
        MA_.write(str(k) + " " + str(seq.annotationClass.map_annotation[1, k, 1] / seq.scale) + "\n")
    LI_.close()
    MA_.close()
# ----------------------------------------------------------------------------------------------------------------------

def save_image(p, seq, patient):
    """ Saves image for visual inspection. """

    path = os.path.join(p.PATH_WALL_SEGMENTATION_RES, "IMAGES_IMC")
    check_dir(path)

    img = np.zeros(seq.firstFrame.shape + (3,))
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = seq.firstFrame, seq.firstFrame, seq.firstFrame

    for k in range(seq.annotationClass.borders['leftBorder'], seq.annotationClass.borders['rightBorder'] + 1, 1):
        img[round(seq.annotationClass.map_annotation[1, k, 0] / seq.scale), k, 2]=150
        img[round(seq.annotationClass.map_annotation[1, k, 0] / seq.scale), k, 0] = 0
        img[round(seq.annotationClass.map_annotation[1, k, 0] / seq.scale), k, 1] = 0
        img[round(seq.annotationClass.map_annotation[1, k, 1] / seq.scale), k, 0] = 150
        img[round(seq.annotationClass.map_annotation[1, k, 1] / seq.scale), k, 1] = 0
        img[round(seq.annotationClass.map_annotation[1, k, 1] / seq.scale), k, 2] = 0

    plt.imsave(os.path.join(path, patient.split('.')[0] + ".png"), img.astype(np.uint8))
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # --- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])

    # --- we get parameters
    p = param.setParameters()

    # --- get image name
    patient_name_list = os.listdir(p.PATH_TO_SEQUENCES)

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
              "overlapping": p.OVERLAPPING}

    wandb.init(project="caroSegDeepInference", entity="nl37", dir=p.PATH_WALL_SEGMENTATION_RES, config=config, name='overlapping_8_fold0')
    wandb.init = (config)

    # --- store metrics
    LI_MAE = np.zeros(0, dtype=np.float32)
    MA_MAE = np.zeros(0, dtype=np.float32)
    IMT_MAE = np.zeros(0, dtype=np.float32)

    # --- launch process
    incr = 0
    for patientName in patient_name_list:
        # patientName = 'tech_044.tiff'
        # if patientName != 'tech_044.tiff' and patientName != 'tech_095.tiff' and patientName != 'tech_042.tiff':
        incr += 1
        print(f' ########### PROGRESSON {incr} / {len(patient_name_list)} ########### ')
        print("Current processed patient: ", patientName)

        # --- path to the data
        path_seq = os.path.join(p.PATH_TO_SEQUENCES, patientName)
        path_borders = os.path.join(p.PATH_TO_BORDERS, patientName.split('.')[0] + "_borders.mat")

        # --- create the object sequenceClass
        seq = sequenceClassIMC(sequence_path =path_seq, path_borders=path_borders, patient_name=patientName, p=p)

        # --- launch the segmentation
        t = time.time()
        seq.sliding_window_vertical_scan()
        elapsed = time.time() - t

        # --- save execution timer and number of patches
        exec_time.write(str(elapsed) + "\n")
        nb_patches.write(str(len(seq.predictionClass.patches)) + "\n")

        # --- save segmentation results
        save_seg(p, seq, patientName)

        # --- save image with LI/MA segmentation
        save_image(p, seq, patientName)

        # --- compute metrics
        MAE_IMT_, MAE_LI_, MAE_MA_ = compute_metrics(p.PATH_LO_LOAD_GT, patientName, 'A1', seq.annotationClass.map_annotation[1,] / seq.scale, p)
        LI_MAE = np.concatenate((LI_MAE, MAE_LI_))
        MA_MAE = np.concatenate((MA_MAE, MAE_MA_))
        IMT_MAE = np.concatenate((IMT_MAE, MAE_IMT_))

        wandb.log({"Patient": patientName.split('.')[0],
                   "MAE_IMT": np.mean(MAE_IMT_),
                   "MAE_LI": np.mean(MAE_LI_),
                   "MAE_MA": np.mean(MAE_MA_),
                   "nb_patches": len(seq.predictionClass.patches),
                   "exec_time": elapsed})

    wandb.log({"MAE_IMT_full": np.mean(IMT_MAE),
               "MAE_LI_full": np.mean(LI_MAE),
               "MAE_MA_full": np.mean(IMT_MAE)})

    exec_time.close()
    nb_patches.close()

    # ----------------------------------------------------------------------------------------------------------------------