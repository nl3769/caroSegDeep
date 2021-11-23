import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import cv2

from medpy.metric.binary import dc, hd


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from caroSegDeepBuildModel.KerasSegmentationFunctions.losses import dice_bce_loss, binary_cross_entropy, dice_loss, dice_bce_constraint_MAE, dice_bce_constraint_thickness
from caroSegDeepBuildModel.KerasSegmentationFunctions.utils import plotOrgGtPred
from caroSegDeepBuildModel.KerasSegmentationFunctions.metrics import iou, dice_coef
from caroSegDeepBuildModel.functionsCaroSeg.model_selection import *
from caroSegDeepBuildModel.classKeras.data_generator import DataGenerator1Channel

# ----------------------------------------------------------------
def test(p, dataset):

    '''
    Infer the model on testing data and compute the DICE and the Hausdorff distance.

    Parameters:
        p:                  object describing all the parameters of the experiment, for reproducibility
        dataset (string): train/test/validation set.
    Returns:
        None
    '''

    # --- get dataset
    data = h5py.File(os.path.join(p.PATH_TO_DATASET), 'r')

    # --- get the dimension of an images for parameters
    dim_img = data[dataset]["img"][list(data[dataset]["img"].keys())[0]][()].shape

    # --- parameters for generator
    params_test = {'dim': dim_img + (1,),
                 'batch_size': 16, # (TODO: param) --> To be discussed as it is not important to change this parameter (not training).
                 'shuffle': False}

    # --- test generator
    test_generator = DataGenerator1Channel(partitions=data[dataset]["img"],
                                          labels=data[dataset]["masks"],
                                          dataAugmentation=False,
                                          **params_test)

    model_filename = os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, 
                                  p.NAME_OF_THE_EXPERIMENT, p.MODEL_SELECTION + '.h5')

    model = ModelSelection(ModelName=p.MODEL_SELECTION,
                           inputShape=(512, 128, 1))
    model.load_weights(model_filename)
    
    # --- display the network
    model.summary()
    
    # --- compile the network
    model.compile(optimizer="adam",
                  loss = globals()[p.LOSS],
                  metrics=[iou, dice_coef])


    # --- Evaluation
    loss_value, iou_value, dice_coef_val = model.evaluate(test_generator, batch_size=1, verbose=1)
    pdf_file_name = os.path.splitext(model_filename)[0] + "_" + dataset + ".pdf"

    datas = data[dataset]["img"]
    GT_datas = data[dataset]["masks"]
    spacing = data[dataset]["spatial_resolution"]
    keys = list(datas.keys())
    batch = 1000
    current_Id = 0

    hausdorff = []
    DICE = []

    condition = True

    while condition==True:

        if current_Id+batch>len(keys):
            batch = len(keys)-1-current_Id
            condition=False

        id = keys[current_Id:current_Id+batch]
        X = np.empty((batch,) + dim_img + (1,), dtype=np.float32)
        y = np.empty((batch,) + dim_img + (1,), dtype=np.float32)
        spatialResolution = np.empty((batch,) + (2,))

        for i, ID in enumerate(id):
            X[i,:,:,0] = datas[ID][()]
            y[i, :, :, 0] = GT_datas[ID][()]
            spatialResolution[i, 0] = spacing[ID][()][0]
            spatialResolution[i, 1] = spacing[ID][()][1]

        y_pred = model.predict(x=X, batch_size=1)
        y_pred[y_pred > 0.5] = 1.
        y_pred[y_pred < 1] = 0.

        for i in range(y_pred.shape[0]):
            ypred = y_pred[i, :, : ,0]
            gt = y[i, :, :, 0]

            if ypred.max() == 0:
                ypred[0,0] = 1
            if gt.max() == 0:
                gt[0,0] = 1

            hausdorff.append(hd(ypred, gt, voxelspacing=(spatialResolution[i][0], spatialResolution[i][1])))
            DICE.append(dc(ypred, gt))

        if current_Id == 0:
            plotOrgGtPred(org=X, gt=y, pred=y_pred, NbImgToPlot=10, figSize=4, OutputPDF=pdf_file_name)

        current_Id += batch


    DICE = np.asarray(DICE)
    SD_DICE = np.std(DICE)
    mean_DICE = np.mean(DICE)

    hausdorff = np.asarray(hausdorff)
    SD_hausdorff = np.std(hausdorff)
    mean_hausdorff = np.mean(hausdorff)

    print("mean DICE: ", mean_DICE, "SD DICE: ", SD_DICE)
    print("mean HD: ", mean_hausdorff, "SD HD: ", SD_hausdorff)


    # Prediction
    WriteMetrics(loss_value = loss_value,
                 iou_value = iou_value,
                 dice_coef_val = dice_coef_val,
                 dice_coef_thresholded = [mean_DICE, SD_DICE],
                 hausforff = [mean_hausdorff, SD_hausdorff],
                 path = os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, 
                                     p.NAME_OF_THE_EXPERIMENT),
                 dataset = dataset)

# ----------------------------------------------------------------
def WriteMetrics(loss_value, iou_value, dice_coef_val, dice_coef_thresholded, hausforff, path, dataset):
    file = open(os.path.join(path, "metrics_results_" + dataset + "_.txt"), "w")

    file.write("Loss on test dataset: " + str(loss_value) + "\n")
    file.write("DICE on test dataset: " + str(dice_coef_val) + "\n")
    file.write("DICE (binary image) test dataset: " + str(dice_coef_thresholded[0]) + ", std: " + str(dice_coef_thresholded[1]) + "\n")
    file.write("Hausdorff distance in mm on test dataset: " + str(hausforff[0]) + ", std: " + str(hausforff[1]) + "\n")
    file.write("IOU on test dataset: " + str(iou_value))

    file.close()
# ----------------------------------------------------------------
def saveImages(x, y, dataSet):

    dim=x.shape

    for k in range(dim[0]):
        img = cv2.hconcat([x[k,:,:,0], y[k,:,:,0]*255])
        plt.imsave('logs/'+dataSet+'/patch_'+str(k)+'.png', img, cmap='gray') # (TODO: param)