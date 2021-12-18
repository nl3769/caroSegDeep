'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import numpy as np
from functions.evaluation import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

# ----------------------------------------------------------------------------------------------------------------------
class evaluationClassIMC():
    ''' evaluationClassIMC contains the code for IMC, mean absolute error and DICE are computed.  '''
    def __init__(self, p):
        self.p=p
    # ------------------------------------------------------------------------------------------------------------------
    def compute_MAE(self):
        ''' Compute the mean absolute error for all patient in the dataset and spread them in train/validation/test. '''
        # --- get predicted patient
        path_to_prediction = os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, "IMC_RES")
        predicted_patients = os.listdir(path_to_prediction)
        for k in range(len(predicted_patients)):
            predicted_patients[k] = predicted_patients[k].split('-')[0]
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
                if patient in predicted_patients:
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
            results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION','IMC_' + set + '_MAE.txt'), "w+")
            results.write("IMC LI_MAE (mean): " + str(np.mean(LI_MAE)) + " um\n")
            results.write("IMC LI_MAE (std): " + str(np.std(LI_MAE)) + " um\n\n\n")
            results.write("IMC MA_MAE (mean): " + str(np.mean(MA_MAE)) + " um\n")
            results.write("IMC MA_MAE (std): " + str(np.std(MA_MAE)) + " um\n\n\n")
            results.write("IMC IMT_MAE (mean): " + str(np.mean(IMT_MAE)) + " um\n")
            results.write("IMC IMT_MAE (std): " + str(np.std(IMT_MAE)) + " um\n\n\n")
            results.close()

            LI_MAE = np.zeros(0, dtype=np.float32)
            MA_MAE = np.zeros(0, dtype=np.float32)
            IMT_MAE = np.zeros(0, dtype=np.float32)

        results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION','IMC_' +  "full_dataset_MAE.txt"), "w+")
        results.write("IMC LI_MAE (mean): " + str(np.mean(LI_MAE_F)) + " um\n")
        results.write("IMC LI_MAE (std): " + str(np.std(LI_MAE_F)) + " um\n\n\n")
        results.write("IMC MA_MAE (mean): " + str(np.mean(MA_MAE_F)) + " um\n")
        results.write("IMC MA_MAE (std): " + str(np.std(MA_MAE_F)) + " um\n\n\n")
        results.write("IMC IMT_MAE (mean): " + str(np.mean(IMT_MAE_F)) + " um\n")
        results.write("IMC IMT_MAE (std): " + str(np.std(IMT_MAE_F)) + " um\n\n\n")
        results.close()
    # ------------------------------------------------------------------------------------------------------------------
    def compute_DICE(self):
        ''' Compute the DICE coefficient for all patient in the dataset and spread them in train/validation/test. '''
        # --- get predicted patient
        path_to_prediction = os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, "IMC_RES")
        predicted_patients = os.listdir(path_to_prediction)
        for k in range(len(predicted_patients)):
            predicted_patients[k] = predicted_patients[k].split('-')[0]
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
                if patient in predicted_patients:
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
                    dice_ = compute_metric_wall_DICE(patient, prediction, expert, borders_ROI, p=self.p)
                    # --- update DICE
                    DICE.append(dice_)
                    DICE_F.append(dice_)
                    # --- increment variable
                    inc += 1

            # --- save results
            results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', 'IMC_' + set + "_DICE.txt"), "w+")
            DICE=np.array(DICE)
            results.write("IMC DICE (mean): " + str(np.mean(DICE)) + "\n")
            results.write("IMC DICE (std): " + str(np.std(DICE)) + "\n\n\n")
            results.close()


        results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', 'IMC_' + "full_dataset_DICE.txt"), "w+")
        DICE_F = np.array(DICE_F)
        results.write("IMC DICE (mean): " + str(np.mean(DICE_F)) + "\n")
        results.write("IMC DICE (std): " + str(np.std(DICE_F)) + "\n\n\n")

        results.close()
    # ------------------------------------------------------------------------------------------------------------------
    def box_plot(self):
        ''' Plot box plot for intima media complex segmentation. '''

        # --- get predicted patient
        path_to_prediction = os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, "IMC_RES")
        predicted_patients = os.listdir(path_to_prediction)
        for k in range(len(predicted_patients)):
            predicted_patients[k] = predicted_patients[k].split('-')[0]
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
        # --- dictionnaries to store results
        MAE_LI={'train': np.empty([0]), 'validation': np.empty([0]), 'test': np.empty([0])}
        MAE_MA = {'train': np.empty([0]), 'validation': np.empty([0]), 'test': np.empty([0])}
        MAE_IMT = {'train': np.empty([0]), 'validation': np.empty([0]), 'test': np.empty([0])}
        for set in sets:
            inc = 0
            for patient in patients[set]:
                if patient in predicted_patients:
                    # --- we get the annotation
                    IFC3_A1, IFC4_A1 = load_annotation(path=self.p.PATH_TO_CONTOURS, patient=patient, expert_name='A1')
                    IFC3_pred, IFC4_pred, borders_pred = load_prediction_IMC(patient, path_to_prediction, IFC3_A1.shape[0])
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
            MAE_LI[set]=LI_MAE
            MAE_MA[set] = MA_MAE
            MAE_IMT[set] = IMT_MAE

            LI_MAE = np.zeros(0, dtype=np.float32)
            MA_MAE = np.zeros(0, dtype=np.float32)
            IMT_MAE = np.zeros(0, dtype=np.float32)

        fig1, ax1 = plt.subplots()
        ax1.set_title('Lumen intima ($MAE$)')
        ax1.boxplot([MAE_LI['train'], MAE_LI['validation'], MAE_LI['test']], showfliers=False)
        plt.xlabel('Subset')
        plt.ylabel('Mean absolute error in $\mu$ m')
        ax1.tick_params(axis='x', colors='red')
        plt.gca().xaxis.set_ticklabels(['train', 'validation', 'test'])

        fig2, ax2 = plt.subplots()
        ax2.set_title('Media adventice ($MAE$)')
        ax2.boxplot([MAE_MA['train'], MAE_MA['validation'], MAE_MA['test']], showfliers=False)
        plt.xlabel('Subset')
        plt.ylabel('Mean absolute error in $\mu$ m')
        ax2.tick_params(axis='x', colors='red')
        plt.gca().xaxis.set_ticklabels(['train', 'validation', 'test'])

        fig3, ax3 = plt.subplots()
        ax3.set_title('Intima media thickness ($MAE$)')
        ax3.boxplot([MAE_IMT['train'], MAE_IMT['validation'], MAE_IMT['test']], showfliers=False)
        plt.xlabel('Subset')
        plt.ylabel('Mean absolute error in $\mu$ m')
        ax3.tick_params(axis='x', colors='red')
        plt.gca().xaxis.set_ticklabels(['train', 'validation', 'test'])


        plt.show()

# ----------------------------------------------------------------------------------------------------------------------
class evaluationClassFW():
    ''' evaluationClassFW contains the code to sort potential initialization failures.  '''
    def __init__(self, p):
        self.p=p

    # ------------------------------------------------------------------------------------------------------------------
    def compute_MAE_FW(self):
        ''' Compute the mean absolute error for all patient in the dataset and spread them in train/validation/test. '''
        # --- get predicted patient
        path_to_prediction = os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, "FAR_WALL_DETECTION")
        predicted_patients = os.listdir(path_to_prediction)
        if '.empty' in predicted_patients:
            predicted_patients.remove('.empty') # add especially for git to create the path
        for k in range(len(predicted_patients)):
            predicted_patients[k] = predicted_patients[k].split('.')[0]
        # --- we read the sets (train/val/test)
        val_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "ValList.txt"))
        test_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "TestList.txt"))
        train_patient = read_fold(os.path.join(self.p.PATH_TO_FOLDS, "TrainList.txt"))

        MEDIAN_MAE_set = np.zeros(0, dtype=np.float32)
        MEDIAN_MAE_full = np.zeros(0, dtype=np.float32)

        sets = ['validation', 'train', 'test']

        patients = {'validation': val_patient,
                    'train': train_patient,
                    'test': test_patient}

        for set in sets:
            for patient in patients[set]:
                if patient in predicted_patients:

                    IFC3, IFC4 = load_annotation(path=self.p.PATH_TO_CONTOURS, patient=patient, expert_name='A1')

                    borders_expert = get_border_expert(IFC3, IFC4)
                    pred, borders_pred = load_prediction_FW(patient, os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'FAR_WALL_DETECTION'))
                    borders_ROI = get_narrow_borders(borders_pred, borders_expert)

                    mae_ = compute_metric_FW_MAE(patient, pred, IFC3=IFC3, IFC4=IFC4, borders=borders_ROI, set=set, p=self.p)
                    MEDIAN_MAE_set = np.concatenate((MEDIAN_MAE_set, mae_))
                    MEDIAN_MAE_full = np.concatenate((MEDIAN_MAE_full, mae_))

                    results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', 'FW_MAE_' + set + ".txt"), "w+")
                    results.write("MEDIAN_MAE (mean): " + str(np.mean(MEDIAN_MAE_set)) + "\n")
                    results.write("MEDIAN_MAE (std): " + str(np.std(MEDIAN_MAE_set)) + "\n\n\n")
                    MEDIAN_MAE_set = np.zeros(0, dtype=np.float32)
                    results.close()

        results = open(os.path.join(self.p.PATH_WALL_SEGMENTATION_RES, "FW__full_dataset_MAE.txt"), "w+")
        results.write("MEDIAN_MAE (mean): " + str(np.mean(MEDIAN_MAE_full)) + "\n")
        results.write("MEDIAN_MAE (std): " + str(np.std(MEDIAN_MAE_full)) + "\n\n\n")
    # ------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------