'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import importlib

from classes.sequence import sequenceClassFW


def save_seg(p, seq, patient):
    ''' Save segmentation results in .txt format. '''
    FW_ = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'FAR_WALL_DETECTION', patient.split('.')[0] + "-LI.txt"), "w+")
    for k in range(seq.annotationClass.borders_ROI['leftBorder'], seq.annotationClass.borders_ROI['rightBorder'] + 1, 1):
        FW_.write(str(k) + " " + str(seq.annotationClass.map_annotation[0   , k, 0] / seq.scale) + "\n")
    FW_.close()

def save_image(p, seq, patient):
    ''' Save image for visual inspection. '''
    img = np.zeros(seq.first_frame.shape + (3,))
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = seq.first_frame, seq.first_frame, seq.first_frame

    for k in range(seq.annotationClass.borders_ROI['leftBorder'], seq.annotationClass.borders_ROI['rightBorder'] + 1, 1):
        img[round(seq.annotationClass.map_annotation[0, k, 0] / seq.scale), k, 2]=0
        img[round(seq.annotationClass.map_annotation[0, k, 0] / seq.scale), k, 0] = 255
        img[round(seq.annotationClass.map_annotation[0, k, 0] / seq.scale), k, 1] = 0

    plt.imsave(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "IMAGES_FW", patient.split('.')[0] + ".png"), img.astype(np.uint8))

if __name__ == '__main__':
    # --- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser=argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg=vars(my_parser.parse_args())
    param=importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])
    # --- we get parameters
    p=param.setParameters()
    # --- get image name
    patient_nameList = os.listdir(p.PATH_TO_SEQUENCES)
    patient_nameList.remove('.empty')    # .empty to add directories to git, do not remove it
    # --- launch process
    for patient_name in patient_nameList:
        print("Current processed patient: ", patient_name)
        # --- path to the data
        path_seq = os.path.join(p.PATH_TO_SEQUENCES, patient_name)
        path_borders = os.path.join(p.PATH_TO_BORDERS, patient_name.split('.')[0] + "_borders.mat")
        # --- create the object sequenceClass
        seq = sequenceClassFW(sequence_path =path_seq, path_borders=path_borders, patient_name=patient_name, p=p)
        # --- launch the segmentation
        seq.launch_seg_far_wall(p=p)
        # --- save segmentation results
        save_seg(p, seq, patient_name)
        # --- save image with far wall delineation
        save_image(p, seq, patient_name)