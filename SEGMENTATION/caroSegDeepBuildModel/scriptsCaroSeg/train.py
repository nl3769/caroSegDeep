'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

import h5py


import datetime


from caroSegDeepBuildModel.KerasSegmentationFunctions.metrics import iou, iou_thresholded, dice_coef
from caroSegDeepBuildModel.KerasSegmentationFunctions.losses import binary_cross_entropy, dice_bce_constraint_MAE, dice_bce_constraint_thickness

from caroSegDeepBuildModel.functionsCaroSeg.save_history import save_IOU, save_loss, save_DICE

from caroSegDeepBuildModel.classKeras.data_generator import DataGenerator1Channel
from caroSegDeepBuildModel.classKeras.custom_callback import CustomLearningRateScheduler, lr_schedule

from caroSegDeepBuildModel.functionsCaroSeg.check_dir import chekDir
from caroSegDeepBuildModel.functionsCaroSeg.model_selection import *


# ----------------------------------------------------------------
def train(p):

    ''' Trains the model from scratch. Training parameters, etc., are defined in set_parameters_training_*.py. '''

    # --- get dataset
    data = h5py.File(p.PATH_TO_DATASET, 'r')
    # --- get the dimension of an images for parameters
    dim_img = data["train"]["img"][list(data["train"]["img"].keys())[0]][()].shape
    # --- parameters for training generator
    params_training = {'dim': dim_img + (1,), 'batch_size': p.BATCH_SIZE, 'shuffle': True}
    # --- parameters for validation generator
    params_val = {'dim': dim_img + (1,), 'batch_size': 1, 'shuffle': False}
    # --- training generator
    training_generator = DataGenerator1Channel(partitions = data["train"]["img"],
                                              labels = data["train"]["masks"],
                                              data_augmentation = p.DATA_AUGMENTATION,
                                              **params_training)
    # --- validation generator
    validation_generator = DataGenerator1Channel(partitions = data["validation"]["img"],
                                                labels = data["validation"]["masks"],
                                                data_augmentation=False,
                                                **params_val)
    # --- path where the results will be saved: figures + model + metrics
    chekDir(os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT))
    model = model_selection(model_name = p.MODEL_SELECTION, input_shape = dim_img + (1,), patch_width = p.PATCH_WIDTH)
    # --- display the model
    model.summary()
    # --- set name the model
    model_filename = os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, p.MODEL_SELECTION + '.h5')
    # --- compile the model
    model.compile(optimizer=Adam(lr = p.LEARNING_RATE), loss = globals()[p.LOSS], metrics=[iou, dice_coef])
    # --- save the model architecture
    plot_model(model, to_file=os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, "model.png"), show_shapes=True)
    # --- path to save tensorboard
    log_dir_tensorboard = os.path.join(p.PATH_TO_SAVE_TENSORBOARD, p.MODEL_SELECTION + "_" + p.NAME_OF_THE_EXPERIMENT + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # --- callbacks
    my_callbacks = [ModelCheckpoint(model_filename, verbose=1, monitor='val_loss', save_best_only=True, save_weights_only=True),
                    EarlyStopping(monitor='val_loss', patience=p.NBPATIENCE_EPOCHS),
                    TensorBoard(log_dir=log_dir_tensorboard),
                    CustomLearningRateScheduler(schedule=lr_schedule)]
    # --- launch training
    history = model.fit(training_generator,
                        validation_data = validation_generator,
                        epochs = p.NB_EPOCH,
                        callbacks = my_callbacks,
                        workers=4) # use four cores instead of one
    # --- save metrics
    save_IOU(history.history['iou'], history.history['val_iou'], os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT), p.MODEL_SELECTION)
    save_loss(history.history['loss'], history.history['val_loss'], os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT), p.MODEL_SELECTION)
    save_DICE(history.history['dice_coef'], history.history['val_dice_coef'], os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT), p.MODEL_SELECTION)
