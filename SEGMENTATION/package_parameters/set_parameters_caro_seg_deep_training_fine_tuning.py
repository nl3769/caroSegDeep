import os
from shutil import copyfile

from package_utils.check_dir                            import chek_dir
from package_parameters.parameters_caro_seg_deep_training import Parameters

def setParameters():

  p = Parameters(

    # --- relative to the training phase
    NB_EPOCH                                          = 300,                                                                # Number of training epochs
    NBPATIENCE_EPOCHS                                 = 50,                                                                # Stop training if validation loss is not improve after NBPATIENCE_EPOCHS epochs
    BATCH_SIZE                                        = 16,                                                                # Batch size
    DATA_AUGMENTATION                                 = True,                                                              # Apply data augmentation on training sets only
    MODEL_SELECTION                                   = 'custom_dilated_unet',                                             # Select architecture
    LEARNING_RATE                                     = 0.0001,                                                          # Learning rate value
    LOSS                                              = 'dice_bce_loss',                                                   # Loss function
    PATH_TO_SAVE_PREDICTION_DURING_TRAINING           = '/home/laine/Documents/ICCVG/FINE_TUNING/PREDICTION_TRAINING',     # One image is predicted at the end of an epoch
    PATCH_HEIGHT                                      = 512,                                                               # The height of a patch
    PATCH_WIDTH                                       = 128,                                                               # The width of a patch

    # --- relative to results
    PATH_TO_SAVE_TENSORBOARD                          = '/home/laine/Documents/PROJECTS_IO/SEGMENTATION/IN_SILICO//'
                                                        'FINE_TUNING/TENSORBOARD/',                                        # path to save tensorboard
    PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS          = '/home/laine/Documents/PROJECTS_IO/SEGMENTATION/IN_SILICO//'
                                                        'FINE_TUNING/RES/',                                                # where metrics/pdf are saved
    NAME_OF_THE_EXPERIMENT                            = 'FINE_TUNING',

    # For fine tuning
    PATH_PRETRAINED_MODEL                             = '/home/laine/cluster/PROJECTS_IO/SEGMENTATION//'
                                                        'CONSORTIUM_MEIBURGER/CARO_SEG_DEEP/RESULTS//'
                                                        'prediction_results/f0_MEIBURGER_01/custom_dilated_unet.h5',      # path to lad pretrained model

    # --- relative to data
    PATH_TO_DATASET                                   = '/home/laine/Documents/PROJECTS_IO/SEGMENTATION//'
                                                        'IN_SILICO/DATASET/SILICO_wall.h5',                               # Path where dataset in .h5 is saved
    PATH_FOLD                                         ={ \
      'training'   : '/home/laine/Documents/PROJECTS_IO/SEGMENTATION/IN_SILICO/DATASET/set/training_patients.txt',
      'validation' : '/home/laine/Documents/PROJECTS_IO/SEGMENTATION/IN_SILICO/DATASET/set/validation_patients.txt',
      'testing'    : '/home/laine/Documents/PROJECTS_IO/SEGMENTATION/IN_SILICO/DATASET/set/testing_patients.txt'}                                                              # Path to fold
  )

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Make copy in results directory in order to track package_parameters
  chek_dir(os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT))
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)),
           os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, 'get_parameters_training.py'))  # we also copy the retained package_parameters in the results
  # --- Return populated object from Parameters class
  return p
