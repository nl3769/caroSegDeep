import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from classes_wall.sequence_semi_auto import sequenceClass
from functions.save_MHA_data import *
import argparse
import importlib

if __name__ == '__main__':

    # --- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])

    p = param.setParameters()

    pathSequence = p.PATH_TO_SEQUENCES
    incr = 1

    if p.PATH_TO_SEQUENCES.split('/')[-1] == 'Sequences':
        nameDB = os.listdir(p.PATH_TO_SEQUENCES)
    elif p.PATH_TO_SEQUENCES.split('/')[-1] == 'Images' or p.PATH_TO_SEQUENCES.split('/')[-1] == 'IMAGES' or p.PATH_TO_SEQUENCES.split('/')[-1] == 'images':
        nameDB = ['']

    patientNameList = os.listdir(p.PATH_TO_SEQUENCES)

    for patientName in patientNameList:
        print("###################################")
        print("Current processed patient: ", patientName)
        print("###################################")

        # --- path to the data
        pathSequence = os.path.join(p.PATH_TO_SEQUENCES, patientName)
        pathBorders = os.path.join(p.PATH_TO_BORDERS, patientName.split('.')[0] + "_borders.mat")

        # --- create the object sequence
        testSequence = sequenceClass(sequencePath = pathSequence,
                                     pathBorders = pathBorders,
                                     patientName=patientName,
                                     p=p)

        # --- launch the segmentation
        testSequence.launch_segmentation_dynamic_vertical_scan()
        # testSequence.computeIMT(p, patientName)

        _LI = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'SEGMENTATION_RES', patientName.split('.')[0] + "-LI.txt"), "w+")
        _MA = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'SEGMENTATION_RES', patientName.split('.')[0] + "-MA.txt"), "w+")

        for k in range(testSequence.annotationClass.borders['leftBorder'], testSequence.annotationClass.borders['rightBorder']+1, 1):
            _LI.write(str(k) + " " + str(testSequence.annotationClass.mapAnnotation[1, k, 0]/testSequence.scale) + "\n")
            _MA.write(str(k) + " " + str(testSequence.annotationClass.mapAnnotation[1, k, 1]/testSequence.scale) + "\n")
        _LI.close()
        _MA.close()


        plt.imsave(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'IMAGES', patientName.split('.')[0] + "-LI.txt"), testSequence.firstFrame, cmap='gray')
        plt.imsave(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'IMAGES', patientName.split('.')[0] + "-LI.txt"), testSequence.predictionClass.mapPrediction[0,], cmap='gray')


