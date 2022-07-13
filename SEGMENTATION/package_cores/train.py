"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

# physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
# tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# os.environ['CUDA_VISIBLE_DEVICES']='0'
import os
import h5py
import datetime

from tensorflow.keras.callbacks                                     import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers                                    import Adam
from tensorflow.keras.callbacks                                     import TensorBoard
from tensorflow.keras.utils                                         import plot_model
from package_utils.metrics                                          import iou, dice_coef
from package_utils.save_history                                     import save_IOU, save_loss, save_DICE
from package_dataset.data_generator                                 import dataGenerator
from package_handler.custom_callback                                import CustomLearningRateScheduler, lr_schedule
from package_utils.check_dir                                        import chek_dir
from package_loss.losses                                            import dice_bce_loss, dice_bce_constraint_MAE
from package_network.model_selection                                import *

# ----------------------------------------------------------------------------------------------------------------------
def train(p):
    """ Trains the model from scratch. Training package_parameters, etc., are defined in set_parameters_training_*.py. """

    # --- get dataset
    data = h5py.File(p.PATH_TO_DATASET, 'r')
    # --- package_parameters for training generator
    params_training = {
        'partitions': data,
        'keys': ["img", "masks"],
        'pfold': p.PATH_FOLD['training'],
        'data_augmentation': p.DATA_AUGMENTATION,
        'batch_size': p.BATCH_SIZE,
        'shuffle': True}
    # --- package_parameters for validation generator
    params_val = {
        'partitions': data,
        'keys': ["img", "masks"],
        'pfold': p.PATH_FOLD['validation'],
        'data_augmentation': False,
        'batch_size': p.BATCH_SIZE,
        'batch_size': 1,
        'shuffle': False}
    # --- training generator
    training_generator = dataGenerator( **params_training)
    # --- validation generator
    validation_generator = dataGenerator(**params_val)
    # --- path where the results will be saved: figures + model + metrics
    chek_dir(os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT))
    dim_img = training_generator.dim
    model = model_selection(model_name=p.MODEL_SELECTION, input_shape=dim_img, patch_width=p.PATCH_WIDTH)
    # --- display the model
    model.summary()
    # --- set name the model
    model_filename = os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, p.MODEL_SELECTION + '.h5')
    # --- compile the model
    model.compile(optimizer=Adam(learning_rate=p.LEARNING_RATE), loss=globals()[p.LOSS], metrics=[iou, dice_coef])
    # --- save the model architecture

    if p.PATH_PRETRAINED_MODEL is not None:
        model.load_weights(p.PATH_PRETRAINED_MODEL)


    plot_model(model, to_file=os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, "model.png"), show_shapes=True)
    # --- path to save tensorboard
    log_dir_tensorboard = os.path.join(p.PATH_TO_SAVE_TENSORBOARD, p.MODEL_SELECTION + "_" + p.NAME_OF_THE_EXPERIMENT + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # --- callbacks
    my_callbacks = [ModelCheckpoint(model_filename, verbose=0, save_best_only=False, save_weights_only=True),
                    #ModelCheckpoint(model_filename, verbose=1, monitor='val_loss', save_best_only=True, save_weights_only=True),
                    EarlyStopping(monitor='val_loss', patience=p.NBPATIENCE_EPOCHS),
                    TensorBoard(log_dir=log_dir_tensorboard),
                    CustomLearningRateScheduler(schedule=lr_schedule)]
    # --- launch training
    history = model.fit(training_generator,
                        validation_data=validation_generator,
                        epochs=p.NB_EPOCH,
                        callbacks=my_callbacks,
                        workers=4)
    # --- save metrics
    save_IOU(history.history['iou'], history.history['val_iou'], os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT), p.MODEL_SELECTION)
    save_loss(history.history['loss'], history.history['val_loss'], os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT), p.MODEL_SELECTION)
    save_DICE(history.history['dice_coef'], history.history['val_dice_coef'], os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT), p.MODEL_SELECTION)