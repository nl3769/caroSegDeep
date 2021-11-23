import matplotlib.pyplot as plt
import os
import sys

import argparse
import importlib

from classes_far_wall.sequence_far_wall import sequenceClass

from parameters.set_parameters import setParameters

if __name__ == '__main__':

    # -- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])

    p = param.setParameters()


    if p.PATH_TO_SEQUENCES.split('/')[-1] == 'Sequences':
        nameDB = os.listdir(p.PATH_TO_SEQUENCES)
    elif p.PATH_TO_SEQUENCES.split('/')[-1] == 'Images':
        nameDB = ['']
    elif p.PATH_TO_SEQUENCES.split('/')[-1] == 'images':
        nameDB = ['']
    elif p.PATH_TO_SEQUENCES.split('/')[-1] == 'IMAGES':
        nameDB = ['']

    for db in nameDB:

        if db == '':
            seq = os.listdir(p.PATH_TO_SEQUENCES)
        else:
            seq = os.listdir(os.path.join(p.PATH_TO_SEQUENCES, db))

        for patientName in seq:

            print(patientName)

            # --- path to the data
            pathAnnotation = os.path.join(p.PATH_TO_CONTOURS, db)
            pathSequence = os.path.join(p.PATH_TO_SEQUENCES, db, patientName)
            pathBorders = os.path.join(p.PATH_TO_BORDERS, db, patientName.split('.')[0] + '_borders.mat')

            # --- create the object sequence
            names = [patientName.split('.')[0] + "_IFC3_A3.mat", patientName.split('.')[0] + "_IFC4_A3.mat"]
            testSequence = sequenceClass(sequencePath = pathSequence,
                                         pathBorders = pathBorders,
                                         patient = patientName,
                                         p = p)

            # --- launch the segmentation
            testSequence.LaunchSegmentationFW(p)

            # --- we save the image with the annotation
            plt.imsave(os.path.join(p.PATH_FAR_WALL_DETECTION_RES, p.FOLD, "IMAGES", patientName.split('.')[0] + ".tiff"), testSequence.firstFrame, cmap='gray')

            # --- we save the annotation in a .txt file
            for k in range(testSequence.annotationClass.bordersROI['leftBorder'], testSequence.annotationClass.borders['leftBorder']):
                testSequence.annotationClass.mapAnnotation[0, k, 0]=0
                testSequence.annotationClass.mapAnnotation[1, k, 0] = 0
            for k in range(testSequence.annotationClass.borders['rightBorder']+1, testSequence.annotationClass.bordersROI['rightBorder']):
                testSequence.annotationClass.mapAnnotation[0, k, 0]=0
                testSequence.annotationClass.mapAnnotation[1, k, 0] = 0

            _res = open(os.path.join(p.PATH_FAR_WALL_DETECTION_RES, p.FOLD, "SEGMENTATION", patientName.split('.')[0] + ".txt"), "w+")
            for k in range(testSequence.annotationClass.mapAnnotation.shape[1]):
                    _res.write(str(k) + " " + str(testSequence.annotationClass.mapAnnotation[0,k,0]/testSequence.scale) + "\n")
            _res.close()



            # plt.imsave('logs/seg/'+patientName+'.png', testSequence.firstFrame, cmap='gray')
