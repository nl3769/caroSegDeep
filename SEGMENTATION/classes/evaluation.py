'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import numpy as np
from functions.evaluation import *

def get_narrow_borders(borders_expert, borders_pred):
    ''' Get the intersection border. '''

    left_border_r=max(borders_expert['left_border'], borders_pred['left_border'])
    right_border_r=min(borders_expert['right_border'], borders_pred['right_border'])

    borders_roi={'left_border': left_border_r,'right_border': right_border_r}

    return borders_roi

class evaluationClassIMC():
    ''' evaluationClass contains the code for evaluation. Nowaday, only mean absolute error is computed.  '''
    def __init__(self, p):
        self.p=p
    # ------------------------------------------------------------------------------------------------------------------
    def compute_MAE(self):
        ''' Compute the mean absolute error for all patient in the dataset and spread them in train/validation/test. '''
        # --- get predicted patient
        path_to_prediction = os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, "IMC_RES")
        predictedPatients = os.listdir(path_to_prediction)
        for k in range(len(predictedPatients)):
            predictedPatients[k] = predictedPatients[k].split('-')[0]
        # --- we read the sets (train/val/test)
        val_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "ValList.txt"))
        test_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "TestList.txt"))
        train_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "TrainList.txt"))
        # --- array to store the results on subset
        LI_MAE = np.zeros(0, dtype=np.float32)
        MA_MAE = np.zeros(0, dtype=np.float32)
        IMT_MAE = np.zeros(0, dtype=np.float32)
        # --- array to store the results on the whole dataset
        LI_MAE_F = np.zeros(0, dtype=np.float32)
        MA_MAE_F = np.zeros(0, dtype=np.float32)
        IMT_MAE_F = np.zeros(0, dtype=np.float32)
        # --- loop over the patients
        sets = ['validation', 'train', 'test']
        patients = {'validation': val_patient, 'train': train_patient, 'test': test_patient}
        for set in sets:
            inc = 0
            for patient in patients[set]:

                if patient in predictedPatients:
                    # --- we get the annotation
                    IFC3_A1, IFC4_A1 = load_annotation(path=self.p.PATH_TO_CONTOURS, patient=patient, expert_name='A1')
                    IFC3_pred, IFC4_pred, borders_pred = load_prediction_IMC(patient, path_to_prediction,
                                                                             IFC3_A1.shape[0])
                    prediction = {'IFC3': IFC3_pred, 'IFC4': IFC4_pred}
                    expert = {'IFC3': IFC3_A1, 'IFC4': IFC4_A1}
                    # --- get the borders
                    borders_expert = get_border_expert(IFC3_A1, IFC4_A1)
                    borders_ROI = get_narrow_borders(borders_pred, borders_expert)
                    # --- we compute the MAE
                    LI, MA, IMT = compute_metric_wall_MAE(patient, prediction, expert, borders_ROI, set=set, p=self.p)
                    # --- we store the MAE
                    LI_MAE = np.concatenate((LI_MAE, LI))
                    MA_MAE = np.concatenate((MA_MAE, MA))
                    IMT_MAE = np.concatenate((IMT_MAE, IMT))

                    LI_MAE_F = np.concatenate((LI_MAE_F, LI))
                    MA_MAE_F = np.concatenate((MA_MAE_F, MA))
                    IMT_MAE_F = np.concatenate((IMT_MAE_F, IMT))
                    # --- increment variable
                    inc += 1

            # --- save results
            results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', set + "_MAE.txt"), "w+")
            results.write("LI_MAE (mean): " + str(np.mean(LI_MAE)) + " um\n")
            results.write("LI_MAE (std): " + str(np.std(LI_MAE)) + " um\n\n\n")
            results.write("MA_MAE (mean): " + str(np.mean(MA_MAE)) + " um\n")
            results.write("MA_MAE (std): " + str(np.std(MA_MAE)) + " um\n\n\n")
            results.write("IMT_MAE (mean): " + str(np.mean(IMT_MAE)) + " um\n")
            results.write("IMT_MAE (std): " + str(np.std(IMT_MAE)) + " um\n\n\n")
            results.close()

            LI_MAE = np.zeros(0, dtype=np.float32)
            MA_MAE = np.zeros(0, dtype=np.float32)
            IMT_MAE = np.zeros(0, dtype=np.float32)

        results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', "full_dataset_MAE.txt"), "w+")
        results.write("LI_MAE (mean): " + str(np.mean(LI_MAE_F)) + " um\n")
        results.write("LI_MAE (std): " + str(np.std(LI_MAE_F)) + " um\n\n\n")
        results.write("MA_MAE (mean): " + str(np.mean(MA_MAE_F)) + " um\n")
        results.write("MA_MAE (std): " + str(np.std(MA_MAE_F)) + " um\n\n\n")
        results.write("IMT_MAE (mean): " + str(np.mean(IMT_MAE_F)) + " um\n")
        results.write("IMT_MAE (std): " + str(np.std(IMT_MAE_F)) + " um\n\n\n")
        results.close()
    # ------------------------------------------------------------------------------------------------------------------
    def compute_DICE(self):
        ''' Compute the DICE coefficient for all patient in the dataset and spread them in train/validation/test. '''
        # --- get predicted patient
        path_to_prediction = os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, "IMC_RES")
        predictedPatients = os.listdir(path_to_prediction)
        for k in range(len(predictedPatients)):
            predictedPatients[k] = predictedPatients[k].split('-')[0]
        # --- we read the sets (train/val/test)
        val_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "ValList.txt"))
        test_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "TestList.txt"))
        train_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "TrainList.txt"))
        # --- array to store the results on the whole dataset
        DICE_F=[]
        # --- loop over the patients
        sets = ['validation', 'train', 'test']
        patients = {'validation': val_patient, 'train': train_patient, 'test': test_patient}
        for set in sets:
            # --- array to store the results on subset
            DICE = []
            inc = 0
            for patient in patients[set]:

                if patient in predictedPatients:
                    # --- we get the annotation
                    IFC3_A1, IFC4_A1 = load_annotation(path=self.p.PATH_TO_CONTOURS, patient=patient, expert_name='A1')
                    IFC3_pred, IFC4_pred, borders_pred = load_prediction_IMC(patient, path_to_prediction,
                                                                             IFC3_A1.shape[0])
                    prediction = {'IFC3': IFC3_pred, 'IFC4': IFC4_pred}
                    expert = {'IFC3': IFC3_A1, 'IFC4': IFC4_A1}
                    # --- get the borders
                    borders_expert = get_border_expert(IFC3_A1, IFC4_A1)
                    borders_ROI = get_narrow_borders(borders_pred, borders_expert)
                    # --- we compute the MAE
                    dice_ = compute_metric_wall_DICE(patient, prediction, expert, borders_ROI, set=set, p=self.p)
                    # --- update DICE
                    DICE.append(dice_)
                    DICE_F.append(dice_)
                    # --- increment variable
                    inc += 1

            # --- save results
            results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', set + "_DICE.txt"), "w+")
            DICE=np.array(DICE)
            results.write("DICE (mean): " + str(np.mean(DICE)) + "\n")
            results.write("DICE (std): " + str(np.std(DICE)) + "\n\n\n")
            results.close()


        results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', "full_dataset_MAE.txt"), "w+")
        DICE_F = np.array(DICE_F)
        results.write("DICE (mean): " + str(np.mean(DICE_F)) + "\n")
        results.write("DICE (std): " + str(np.std(DICE_F)) + "\n\n\n")

        results.close()