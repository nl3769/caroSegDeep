"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import numpy as np
import os
import h5py

from medpy.metric.binary import dc, hd


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from package_utils.utils                    import plot_org_gt_pred
from package_utils.metrics                  import iou, dice_coef
from package_network.model_selection        import *
from package_dataset.data_generator         import dataGenerator
from package_loss.losses                    import dice_bce_loss, dice_bce_constraint_MAE

# ----------------------------------------------------------------
def test(p, set):

    """ Infer the model on testing data and compute the DICE and the Hausdorff distance. """

    # --- get set
    data = h5py.File(os.path.join(p.PATH_TO_DATASET), 'r')
    # --- package_parameters for generator
    params_test = {'partitions': data,
                   'keys': ["img", "masks"],
                   'pfold': os.path.join(p.PATH_FOLD[set]),
                   'data_augmentation': p.DATA_AUGMENTATION,
                   'batch_size': 16,
                   'shuffle': False}
    # --- generator
    test_generator = dataGenerator(**params_test)
    dim_img = test_generator.dim
    # --- name of the model
    model_filename = os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, p.MODEL_SELECTION + '.h5')
    # --- load the model
    model = model_selection(model_name=p.MODEL_SELECTION, input_shape=dim_img)
    model.load_weights(model_filename)
    # --- display the network
    model.summary()
    # --- compile the network
    model.compile(optimizer="adam", loss = globals()[p.LOSS], metrics=[iou, dice_coef])

    # --- Evaluation
    loss_value, iou_value, dice_coef_val = model.evaluate(test_generator, batch_size=1, verbose=1)
    pdf_file_name = os.path.splitext(model_filename)[0] + "_" + set.split('/')[-1].split('.')[0] + ".pdf"

    hausdorff, DICE = compute_hausdorff_binary_dice(dim_img=dim_img, data=data, model=model, pdf_file_name=pdf_file_name)
    DICE = np.asarray(DICE)
    SD_DICE = np.std(DICE)
    mean_DICE = np.mean(DICE)
    hausdorff = np.asarray(hausdorff)
    SD_hausdorff = np.std(hausdorff)
    mean_hausdorff = np.mean(hausdorff)

    # --- save metrics
    write_metrics(loss_value = loss_value,
                  iou_value = iou_value,
                  dice_coef_val = dice_coef_val,
                  dice_coef_thresholded = [mean_DICE, SD_DICE],
                  hausforff = [mean_hausdorff, SD_hausdorff],
                  path = os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT),
                  set = set)

# ----------------------------------------------------------------------------------------------------------------------
def write_metrics(loss_value: float, iou_value: float, dice_coef_val: float, dice_coef_thresholded: float, hausforff: float, path: str, set: str):
    """ Write metrics in txt file. """
    
    with open(os.path.join(path, "metrics_results_" + set + ".txt"), "w") as file:
        file.write("Loss on test set: " + str(loss_value) + "\n")
        file.write("DICE on test set: " + str(dice_coef_val) + "\n")
        file.write("DICE (binary image) test set: " + str(dice_coef_thresholded[0]) + ", std: " + str(dice_coef_thresholded[1]) + "\n")
        file.write("Hausdorff distance in mm on test set: " + str(hausforff[0]) + ", std: " + str(hausforff[1]) + "\n")
        file.write("IOU on test set: " + str(iou_value))


# ----------------------------------------------------------------------------------------------------------------------
def compute_hausdorff_binary_dice(dim_img: tuple, data: list, model, pdf_file_name: str):
    """ Compute the Hausdorff distance and the binarised DICE coefficient on the whole subset. """

    # --- get fnames
    patients = list(data.keys())

    condition=True
    current_id=0
    batch=1
    hausdorff=[]
    DICE=[]

    X = np.empty((batch,) + dim_img, dtype=np.float32)
    y = np.empty((batch,) + dim_img, dtype=np.float32)
    spatial_res = np.empty((batch,) + (2,))

    nb_plot=10
    X_plot = np.empty((nb_plot,) + dim_img, dtype=np.float32)
    Y_plot = np.empty((nb_plot,) + dim_img, dtype=np.float32)
    PRED_plot = np.empty((nb_plot,) + dim_img, dtype=np.float32)
    incr=0
    for patient in patients:
        for patch in data[patient]['img']:

            # --- load data
            X[0, :, :, 0] = data[patient]['img'][patch][()]
            y[0, :, :, 0] = data[patient]['masks'][patch][()]
            spatial_res[0, 0] = data[patient]['spatial_resolution'][patch][()][0]
            spatial_res[0, 1] = data[patient]['spatial_resolution'][patch][()][1]

            # --- predicting the patch and binarising it
            y_pred = model.predict(x=X, batch_size=1)
            y_pred[y_pred > 0.5] = 1.
            y_pred[y_pred < 1] = 0.

            # --- compute hausdorff distance
            ypred = y_pred[0, :, :, 0]
            gt = y[0, :, :, 0]
            if ypred.max() == 0:
                ypred[0, 0] = 1
            if gt.max() == 0:
                gt[0, 0] = 1
            hausdorff.append(hd(ypred, gt, voxelspacing=(spatial_res[0][0], spatial_res[0][1])))
            DICE.append(dc(ypred, gt))

            if incr < nb_plot:
                X_plot[incr, ] = X[0, ]
                Y_plot[incr,] = y[0,]
                PRED_plot[incr,] = y_pred[0,]
                incr+=1
                if incr == nb_plot:
                    plot_org_gt_pred(org=X_plot, gt=Y_plot, pred=PRED_plot, NbImgToPlot=10, figSize=4, OutputPDF=pdf_file_name)
        # if current_id == 0:
        #     plot_org_gt_pred(org=X, gt=y, pred=y_pred, NbImgToPlot=10, figSize=4, OutputPDF=pdf_file_name)
    # while condition == True:
        # --- ensure that the whole subset is processed
        # if current_id + batch > len(patient):
        #     batch = len(keys) - 1 - current_id
        #     condition = False
        # --- get the current batch
        # id=keys[current_id:current_id + batch]
        # X=np.empty((batch,) + dim_img + (1,), dtype=np.float32)
        # y=np.empty((batch,) + dim_img + (1,), dtype=np.float32)
        # spatial_res = np.empty((batch,) + (2,))
        # for i, ID in enumerate(id):
        #     X[i, :, :, 0] = data[ID][()]
        #     y[i, :, :, 0] = GT_data[ID][()]
        #     spatial_res[i, 0] = spacing[ID][()][0]
        #     spatial_res[i, 1] = spacing[ID][()][1]
        # # --- predicting the patch and binarising it
        # y_pred = model.predict(x=X, batch_size=1)
        # y_pred[y_pred > 0.5] = 1.
        # y_pred[y_pred < 1] = 0.
        # # --- compute the hausdorff distance and the binarized DICE coefficient on the whole batch
        # for i in range(y_pred.shape[0]):
        #     ypred = y_pred[i, :, :, 0]
        #     gt = y[i, :, :, 0]
        #     if ypred.max() == 0:
        #         ypred[0, 0] = 1
        #     if gt.max() == 0:
        #         gt[0, 0] = 1
        #     hausdorff.append(hd(ypred, gt, voxelspacing=(spatial_res[i][0], spatial_res[i][1])))
        #     DICE.append(dc(ypred, gt))
        # if current_id == 0:
        #     plot_org_gt_pred(org=X, gt=y, pred=y_pred, NbImgToPlot=10, figSize=4, OutputPDF=pdf_file_name)
        #
        # current_id+=batch

    return hausdorff, DICE
