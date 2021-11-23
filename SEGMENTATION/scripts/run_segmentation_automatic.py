import matplotlib.pyplot as plt
import os
from common.sequence_automatic import sequenceClass
import argparse
import importlib
from functions.save_MHA_data import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # --- using a parser with set_parameters.py allows us to run several process with different set_parameters.py with the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('parameters.' + arg['Parameters'].split('.')[0])

    p = param.setParameters()
    listPatient = os.listdir(p.PATH_TO_SEQUENCES)

    for patientName in listPatient:

        print(patientName)

        # --- path to the data
        pathAnnotation = p.PATH_TO_CONTOURS
        pathSequence = os.path.join(p.PATH_TO_SEQUENCES, patientName)
        pathBorders = os.path.join(p.PATH_TO_BORDERS, patientName.split('.')[0] + "_borders.mat")

        # --- create the object sequence
        names = [patientName.split('.')[0] + "_IFC3_A3.mat", patientName.split('.')[0] + "_IFC4_A3.mat"]


        testSequence = sequenceClass(sequencePath = pathSequence,
                                   pathBorders = pathBorders,
                                   p=p,
                                   patientName=patientName)

        testSequence.launchSegmentation_dynamic_vertical_scan()


        # saveExpert(patient=patientName,
        #            pathToAnnotation=pathAnnotation,
        #            p=p)
        #
        # pathToSave = p.PATH_TO_SAVE_RESULTS_SEGMENTATION
        # _LI = open(os.path.join(pathToSave, p.FOLD, patientName.split('.')[0] + "-LI.txt"), "w+")
        # _MA = open(os.path.join(pathToSave, p.FOLD, patientName.split('.')[0] + "-MA.txt"), "w+")
        #
        # for k in range(testSequence.annotationClass.bordersROI['leftBorder'],
        #                testSequence.annotationClass.bordersROI['rightBorder'] + 1, 1):
        #     _LI.write(str(k) + " " + str(testSequence.annotationClass.mapAnnotation[1, k, 0]) + "\n")
        #     _MA.write(str(k) + " " + str(testSequence.annotationClass.mapAnnotation[1, k, 1]) + "\n")
        #
        # _LI.close()
        # _MA.close()

