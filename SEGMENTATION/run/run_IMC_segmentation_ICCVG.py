"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import numpy                    as np
import matplotlib.pyplot        as plt
import os
import argparse
import importlib
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from package_handler.sequence import sequenceClassIMC

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
    """ Save segmentation results in .txt format. """

    path = os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'IMC_RES')
    check_dir(path)


    for id in range(seq.annotationClass.map_annotation.shape[0]-1):

        if id < 10:
            id_name = '00' + str(id)

        if (id < 100) and (id >= 10):
            id_name = '0' + str(id)

        if id >= 100 :
            id_name = str(id)

        LI_ = open(os.path.join(path, patient.split('.')[0] + "-id_" + id_name + "-LI.txt"), "w+")
        MA_ = open(os.path.join(path, patient.split('.')[0] + "-id_" + id_name + "-MA.txt"), "w+")
        for k in range(seq.annotationClass.borders_ROI['leftBorder'], seq.annotationClass.borders_ROI['rightBorder'], 1):
            LI_.write(str(k) + " " + str(seq.annotationClass.map_annotation[id+1, k, 0] / seq.scale) + "\n")
            MA_.write(str(k) + " " + str(seq.annotationClass.map_annotation[id+1, k, 1] / seq.scale) + "\n")

        LI_.close()
        MA_.close()

# ----------------------------------------------------------------------------------------------------------------------
def save_image(p, seq, patient):
    """ Saves image for visual inspection. """

    path = os.path.join(p.PATH_WALL_SEGMENTATION_RES, "IMAGES_IMC")
    check_dir(path)

    img = np.zeros(seq.firstFrame.shape[1:] + (3,))

    for id in range(seq.annotationClass.map_annotation.shape[0] - 1):

        img[:, :, 0], img[:, :, 1], img[:, :, 2] = seq.firstFrame[id,], seq.firstFrame[id,], seq.firstFrame[id,]
        if id < 10:
            id_name = '00' + str(id)
        if (id >= 10) and (id < 100) :
            id_name = '0' + str(id)
        if id >= 100:
            id_name = str(id)

        for k in range(seq.annotationClass.borders_ROI['leftBorder'], seq.annotationClass.borders_ROI['rightBorder'] + 1, 1):

            img[round(seq.annotationClass.map_annotation[id+1, k, 0] / seq.scale), k, 0] = 0
            img[round(seq.annotationClass.map_annotation[id+1, k, 0] / seq.scale), k, 1] = 0
            img[round(seq.annotationClass.map_annotation[id+1, k, 0] / seq.scale), k, 2] = 150

            img[round(seq.annotationClass.map_annotation[id+1, k, 1] / seq.scale), k, 0] = 150
            img[round(seq.annotationClass.map_annotation[id+1, k, 1] / seq.scale), k, 1] = 0
            img[round(seq.annotationClass.map_annotation[id+1, k, 1] / seq.scale), k, 2] = 0

        plt.imsave(os.path.join(path, patient.split('.')[0] + "-id_" + id_name + ".png"), img.astype(np.uint8))
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # --- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of package_parameters required to execute the code.')
    my_parser.add_argument('--fold', '-fold', type=str)
    # my_parser.add_argument('--overlap', '-overlap', type=int)

    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])

    # --- we get package_parameters
    p = param.setParameters()

    # --- we get package_parameters
    p.PATH_TO_LOAD_TRAINED_MODEL_WALL = os.path.join(p.PATH_TO_LOAD_TRAINED_MODEL_WALL + arg['fold'].split('f')[-1], 'EXAMPLE')

    # --- get image name
    patient_name_list = os.listdir(p.PATH_TO_SEQUENCES)
    patient_name_list = [patient for patient in patient_name_list if '.nii' in patient]
    # patient_name_list = get_patient_name(p.PATH_TO_FOLDS, patient_name_list)
    # --- create exec_time obj
    check_dir(p.PATH_WALL_SEGMENTATION_RES)
    if os.path.isfile(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "exec_time.txt")):
        os.remove(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "exec_time.txt"))
    if os.path.isfile(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "nb_patches.txt")):
        os.remove(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "nb_patches.txt"))
    exec_time = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "exec_time.txt"), "w")
    nb_patches = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "nb_patches.txt"), "w")

    # --- store metrics
    LI_MAE = np.zeros(0, dtype=np.float32)
    MA_MAE = np.zeros(0, dtype=np.float32)
    IMT_MAE = np.zeros(0, dtype=np.float32)

    # --- launch process
    incr = 0

    for patientName in patient_name_list:

        incr += 1
        print(f' ########### PROGRESSION {incr} / {len(patient_name_list)} ########### ')
        print("Current processed patient: ", patientName)

        # --- path to the data
        path_seq = os.path.join(p.PATH_TO_SEQUENCES, patientName)
        path_borders = os.path.join(p.PATH_TO_BORDERS, patientName.split('.')[0] + "_borders.mat")

        # --- create the object sequenceClass
        seq = sequenceClassIMC(sequence_path =path_seq, path_borders=path_borders, patient_name=patientName, p=p)

        # --- launch the segmentation
        t = time.time()
        postprocess_time = seq.sliding_window_vertical_scan()
        elapsed = time.time() - t

        # --- save execution timer and number of patches
        exec_time.write(str(elapsed) + "\n")
        nb_patches.write(str(len(seq.predictionClass.patches)) + "\n")

        # --- save segmentation results
        save_seg(p, seq, patientName)

        # --- save image with LI/MA segmentation
        save_image(p, seq, patientName)

        # --- compute metrics
        # MAE_IMT_, MAE_LI_, MAE_MA_ = compute_metrics(p.PATH_LO_LOAD_GT, patientName, 'A1', seq.annotationClass.map_annotation[1,] / seq.scale, p)
        # LI_MAE = np.concatenate((LI_MAE, MAE_LI_))
        # MA_MAE = np.concatenate((MA_MAE, MAE_MA_))
        # IMT_MAE = np.concatenate((IMT_MAE, MAE_IMT_))


    exec_time.close()
    nb_patches.close()

    # ----------------------------------------------------------------------------------------------------------------------