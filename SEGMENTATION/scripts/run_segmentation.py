import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import importlib

from classes.sequence import sequenceClass


def save_seg(p, seq, patient):
    '''
    save segmenation results in .txt format
    '''
    _LI = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'SEGMENTATION_RES', patient.split('.')[0] + "-LI.txt"), "w+")
    _MA = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'SEGMENTATION_RES', patient.split('.')[0] + "-MA.txt"), "w+")

    for k in range(seq.annotationClass.borders['leftBorder'], seq.annotationClass.borders['rightBorder'] + 1, 1):
        _LI.write(str(k) + " " + str(seq.annotationClass.mapAnnotation[1, k, 0] / seq.scale) + "\n")
        _MA.write(str(k) + " " + str(seq.annotationClass.mapAnnotation[1, k, 1] / seq.scale) + "\n")

    _LI.close()
    _MA.close()

def save_image(p, seq, patient):
    '''
    save image for visual inspection
    '''
    img = np.zeros(seq.firstFrame.shape + (3,))
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = seq.firstFrame, seq.firstFrame, seq.firstFrame

    for k in range(seq.annotationClass.borders['leftBorder'], seq.annotationClass.borders['rightBorder'] + 1, 1):
        img[round(seq.annotationClass.mapAnnotation[1, k, 0] / seq.scale), k, 2]=150
        img[round(seq.annotationClass.mapAnnotation[1, k, 0] / seq.scale), k, 0] = 0
        img[round(seq.annotationClass.mapAnnotation[1, k, 0] / seq.scale), k, 1] = 0

        img[round(seq.annotationClass.mapAnnotation[1, k, 1] / seq.scale), k, 0] = 150
        img[round(seq.annotationClass.mapAnnotation[1, k, 1] / seq.scale), k, 1] = 0
        img[round(seq.annotationClass.mapAnnotation[1, k, 1] / seq.scale), k, 2] = 0
        # _LI.write(str(k) + " " + str(seq.annotationClass.mapAnnotation[1, k, 0] / seq.scale) + "\n")
        # _MA.write(str(k) + " " + str(seq.annotationClass.mapAnnotation[1, k, 1] / seq.scale) + "\n")

    plt.imsave(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "IMAGES", patient.split('.')[0] + ".png"), img.astype(np.uint8))


if __name__ == '__main__':

    # --- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser=argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg=vars(my_parser.parse_args())
    param=importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])
    p=param.setParameters()

    # --- get image name
    patientNameList = os.listdir(p.PATH_TO_SEQUENCES)
    patientNameList.remove('.empty')    # .empty to add directories to the git, do not remove it

    # --- launch process
    for patientName in patientNameList:

        print("Current processed patient: ", patientName)

        # --- path to the data
        pathSequence = os.path.join(p.PATH_TO_SEQUENCES, patientName)
        pathBorders = os.path.join(p.PATH_TO_BORDERS, patientName.split('.')[0] + "_borders.mat")

        # --- create the object sequence
        seq = sequenceClass(sequencePath = pathSequence,
                            pathBorders = pathBorders,
                            patientName=patientName,
                            p=p)

        # --- launch the segmentation
        seq.launch_segmentation_dynamic_vertical_scan()
        # --- save segmentation results
        save_seg(p, seq, patientName)
        # --- save image with LI/MA segmentation
        save_image(p, seq, patientName)