import numpy as np
import scipy.io
import os
import cv2
import matplotlib.pyplot as plt

def read_fold(path):

    patientsN = open(path, "r")
    patients = patientsN.readlines()
    patientsN.close()

    for k in range(len(patients)):
        patients[k] = patients[k].split('.')[0]

    return patients
# ----------------------------------------------------------------------------------------------------------------------
def load_prediction_IMC(patient_name, path, img_width):
    ''' Load IMC prediction. '''

    IFC3_pred = open(os.path.join(path, patient_name + "-LI.txt"), "r")
    IFC3 = IFC3_pred.readlines()
    IFC3_pred.close()

    IFC4_pred = open(os.path.join(path, patient_name + "-MA.txt"), "r")
    IFC4 = IFC4_pred.readlines()
    IFC4_pred.close()

    for k in range(len(IFC4)):
        if float(IFC4[k].split(' ')[1].replace("\n", ""))!=0:
            left_border = int(IFC4[k].split(' ')[0])
            break

    for k in range(len(IFC4)-1,0,-1):
        if float(IFC4[k].split(' ')[1].replace("\n", ""))!=0:
            right_border = int(IFC4[k].split(' ')[0])
            break

    IFC3_np, IFC4_np = np.zeros(img_width), np.zeros(img_width)

    for k in range(left_border, right_border+1):
        IFC3_np[k] = IFC3[k-left_border].split('\n')[0].split(' ')[-1]
        IFC4_np[k] = IFC4[k-left_border].split('\n')[0].split(' ')[-1]

    return IFC3_np, IFC4_np, {'left_border': left_border, 'right_border': right_border}
# ----------------------------------------------------------------------------------------------------------------------
def load_annotation(path, patient, expert_name):
    ''' Load experts' annotation. '''
    IFC3 = scipy.io.loadmat(os.path.join(path, expert_name, patient + "_IFC3_" + expert_name + ".mat"))['seg'][:,0]
    IFC4 = scipy.io.loadmat(os.path.join(path, expert_name, patient + "_IFC4_" + expert_name + ".mat"))['seg'][:,0]

    return IFC3, IFC4
# ----------------------------------------------------------------------------------------------------------------------
def read_CF_directory(path):
    '''Read calibration factor. '''
    f = open(path, "r")
    val = f.readline().split(' \n')
    return float(val[0])
# ----------------------------------------------------------------------------------------------------------------------
def get_border_expert(IFC3, IFC4):

    for k in range(IFC3.shape[0]):
        if IFC3[k]!=0:
            left_border_3=k
            break
    for k in range(IFC3.shape[0]-1, 0, -1):
        if IFC3[k]!=0:
            right_border_3=k
            break

    for k in range(IFC4.shape[0]):
        if IFC4[k]!=0:
            left_border_4=k
            break
    for k in range(IFC4.shape[0]-1, 0, -1):
        if IFC4[k]!=0:
            right_border_4=k
            break

    if left_border_4 > left_border_3:
        left_border = left_border_4
    else:
        left_border = left_border_3

    if right_border_4 < right_border_3:
        right_border = right_border_4
    else:
        right_border = right_border_3

    return {'left_border': left_border, 'right_border': right_border}
# ----------------------------------------------------------------------------------------------------------------------
def compute_metric_wall_MAE(patient: str, prediction: dict, expert: dict, borders_ROI: dict, set: str, p, save_outlier= False):

    scale = read_CF_directory(os.path.join(p.PATH_TO_CF, patient + "_CF.txt"))

    IFC3_pred=prediction['IFC3']
    IFC4_pred = prediction['IFC4']

    IFC3_exp=expert['IFC3']
    IFC4_exp = expert['IFC4']

    LI_MAE = np.abs(IFC3_exp[borders_ROI['left_border']:borders_ROI['right_border']+1]-\
                    IFC3_pred[borders_ROI['left_border']:borders_ROI['right_border']+1])*scale*1000
    MA_MAE = np.abs(IFC4_exp[borders_ROI['left_border']:borders_ROI['right_border']+1] - \
                    IFC4_pred[borders_ROI['left_border']:borders_ROI['right_border']+1])*scale*1000
    IMT_MAE = np.abs((IFC4_exp[borders_ROI['left_border']:borders_ROI['right_border']+1]-\
                      IFC3_exp[borders_ROI['left_border']:borders_ROI['right_border']+1])-\
                     (IFC4_pred[borders_ROI['left_border']:borders_ROI['right_border']+1]-\
                      IFC3_pred[borders_ROI['left_border']:borders_ROI['right_border']+1]))*scale*1000

    ind_mean_LI_MAE = np.mean(LI_MAE)
    ind_mean_MA_MAE = np.mean(MA_MAE)
    ind_mean_IMT_MAE = np.mean(IMT_MAE)

    # --- we display potential outliers
    tresh = 250
    if (ind_mean_LI_MAE > tresh or ind_mean_MA_MAE > tresh or ind_mean_IMT_MAE > tresh) and save_outlier:
        img = cv2.imread(os.path.join(p.PATH_TO_SEQUENCES, patient + ".tiff"))

        for k in range(borders_ROI['left_border'], borders_ROI['right_border']+1):
            img[round(IFC3_pred[k]), k, 0] = 0
            img[round(IFC3_pred[k]), k, 1] = 0
            img[round(IFC3_pred[k]), k, 2] = 255

            img[round(IFC4_pred[k]), k, 0] = 0
            img[round(IFC4_pred[k]), k, 1] = 0
            img[round(IFC4_pred[k]), k, 2] = 255

        for k in range(borders_ROI['left_border'], borders_ROI['right_border'] + 1):
            img[round(IFC3_exp[k]), k, 0] = 0
            img[round(IFC3_exp[k]), k, 1] = 255
            img[round(IFC3_exp[k]), k, 2] = 0

            img[round(IFC4_exp[k]), k, 0] = 0
            img[round(IFC4_exp[k]), k, 1] = 255
            img[round(IFC4_exp[k]), k, 2] = 0

        cv2.imwrite(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "EVALUATION", patient + ".jpg"), img)

        err = open(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', set + '_' + patient + ".txt"), 'w+')
        err.write("Lumen intima: " + str(ind_mean_LI_MAE) + "\n")
        err.write("Media adventice: " + str(ind_mean_MA_MAE) + "\n")
        err.write("intima media: " + str(ind_mean_IMT_MAE) + "\n")
        err.close

    return LI_MAE, MA_MAE, IMT_MAE
# ----------------------------------------------------------------------------------------------------------------------
def compute_metric_wall_DICE(patient: str, prediction: dict, expert: dict, borders_ROI: dict, p):

    img = cv2.imread(os.path.join(p.PATH_TO_SEQUENCES, patient + ".tiff"))
    dim_img=img.shape[0:2]
    GT_mask, pred_mask=np.zeros(dim_img), np.zeros(dim_img)

    # --- mask GT and predicted mask
    for k in range(borders_ROI['left_border'], borders_ROI['right_border']):
        pred_mask[round(expert['IFC3'][k]):round(expert['IFC4'][k]), k] = 1
        GT_mask[round(prediction['IFC3'][k]):round(prediction['IFC4'][k]), k] = 1

    # --- compute DICE coefficient
    intersection=np.sum(pred_mask*GT_mask)
    sum_pred_mask=np.sum(pred_mask)
    sum_GT_mask=np.sum(GT_mask)
    dice=2*(intersection+np.finfo(float).eps)/(sum_pred_mask+sum_GT_mask)

    print("DICE: ", dice)

    return dice
# ----------------------------------------------------------------------------------------------------------------------
def compute_metric_FW_MAE(patient, pred, IFC3, IFC4, borders, set, p):


    GT = (IFC3+IFC4)/2
    scale = read_CF_directory(os.path.join(p.PATH_TO_CF, patient + "_CF.txt"))
    left_border=borders['left_border']
    right_border=borders['right_border']

    metric = np.abs(GT[left_border:right_border] - pred[left_border:right_border])*scale*1000
    # --- 800um -> just a distance far enough from the mean value. If the MAE is greater than 800 then it can be an outlier
    if metric.max() > 800:

        img = cv2.imread(os.path.join(p.PATH_TO_SEQUENCES, patient + ".tiff"))
        for k in range(left_border, right_border+1):

            img[round(pred[k]), k, 0], img[round(pred[k]), k, 1], img[round(pred[k]), k, 2] = 255, 255, 255

            img[round(IFC4[k]), k, 0] = 255
            img[round(IFC4[k]), k, 1], img[round(IFC4[k]), k, 2] = 0, 0

            img[round(IFC3[k]), k, 1] = 255
            img[round(IFC3[k]), k, 2], img[round(IFC3[k]), k, 0] = 0, 0

        cv2.imwrite(os.path.join(p.PATH_WALL_SEGMENTATION_RES, 'EVALUATION', "FW_OUTLIERS", set + '_' + patient + ".jpg"), img)

        # --- uncomment to save the metrics; but only the image is useful for visual inspection.
        # err = open(os.path.join(p.PATH_TO_RES, , patient + ".txt"), 'w+')
        # err.write("max distance by column: " + str(metric.max()) + "\n")
        # err.write("average distance by column: " + str(metric.mean()) + "\n")
        # err.write("std average distance by column: " + str(metric.std()) + "\n")
        # err.close

    return metric
# ----------------------------------------------------------------------------------------------------------------------
def load_prediction_FW(patientName: str, path: str):

    predN = open(os.path.join(path, patientName + ".txt"), "r")
    prediction = predN.readlines()
    predN.close()

    pred = np.zeros(len(prediction))

    for k in range(len(prediction)):
        pred[k] = prediction[k].split('\n')[0].split(' ')[-1]
        if k==0:
            left_border=int(prediction[k].split('\n')[0].split(' ')[0])
        if k==len(prediction)-1:
            right_border=int(prediction[k].split('\n')[0].split(' ')[0])
    pred=np.concatenate((np.zeros(left_border), pred))
    return pred, {'left_border': left_border, 'right_border': right_border}
# ----------------------------------------------------------------------------------------------------------------------
def get_narrow_borders(borders_expert, borders_pred):
    ''' Get the intersection border. '''

    left_border_r=max(borders_expert['left_border'], borders_pred['left_border'])
    right_border_r=min(borders_expert['right_border'], borders_pred['right_border'])

    borders_roi={'left_border': left_border_r,'right_border': right_border_r}

    return borders_roi