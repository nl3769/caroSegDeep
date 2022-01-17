'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import importlib
import time

from classes.sequence import sequenceClassIMC


def save_seg(p, seq, patient):
    ''' Save segmentation results in .txt format. '''
    LI_ = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'IMC_RES', patient.split('.')[0] + "-LI.txt"), "w+")
    MA_ = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'IMC_RES', patient.split('.')[0] + "-MA.txt"), "w+")
    for k in range(seq.annotationClass.borders['leftBorder'], seq.annotationClass.borders['rightBorder'] + 1, 1):
        LI_.write(str(k) + " " + str(seq.annotationClass.map_annotation[1, k, 0] / seq.scale) + "\n")
        MA_.write(str(k) + " " + str(seq.annotationClass.map_annotation[1, k, 1] / seq.scale) + "\n")
    LI_.close()
    MA_.close()

def save_image(p, seq, patient):
    ''' Saves image for visual inspection. '''
    img = np.zeros(seq.firstFrame.shape + (3,))
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = seq.firstFrame, seq.firstFrame, seq.firstFrame

    for k in range(seq.annotationClass.borders['leftBorder'], seq.annotationClass.borders['rightBorder'] + 1, 1):
        img[round(seq.annotationClass.map_annotation[1, k, 0] / seq.scale), k, 2]=150
        img[round(seq.annotationClass.map_annotation[1, k, 0] / seq.scale), k, 0] = 0
        img[round(seq.annotationClass.map_annotation[1, k, 0] / seq.scale), k, 1] = 0
        img[round(seq.annotationClass.map_annotation[1, k, 1] / seq.scale), k, 0] = 150
        img[round(seq.annotationClass.map_annotation[1, k, 1] / seq.scale), k, 1] = 0
        img[round(seq.annotationClass.map_annotation[1, k, 1] / seq.scale), k, 2] = 0

    plt.imsave(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "IMAGES_IMC", patient.split('.')[0] + ".png"), img.astype(np.uint8))

if __name__ == '__main__':

    # --- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser=argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg=vars(my_parser.parse_args())
    param=importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])

    # --- we get parameters
    p=param.setParameters()

    # --- get image name
    patient_name_list = os.listdir(p.PATH_TO_SEQUENCES)
    patient_name_list.remove('.empty')    # .empty to add directories to git, do not remove it

    # --- create exec_time obj
    if os.path.isfile(os.path.join(p.PATH_EXECUTION_TIME, "exec_time.txt")):
        os.remove(os.path.join(p.PATH_EXECUTION_TIME, "exec_time.txt"))
    if os.path.isfile(os.path.join(p.PATH_EXECUTION_TIME, "nb_patches.txt")):
        os.remove(os.path.join(p.PATH_EXECUTION_TIME, "nb_patches.txt"))
    exec_time = open(os.path.join(p.PATH_EXECUTION_TIME, "exec_time.txt"), "w")
    nb_patches = open(os.path.join(p.PATH_EXECUTION_TIME, "nb_patches.txt"), "w")

    # --- launch process
    incr=0
    for patientName in patient_name_list:
        incr+=1
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

    exec_time.close()
    nb_patches.close()