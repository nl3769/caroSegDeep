import os

from package_parameters.parameters_far_wall_deep_training import Parameters
from shutil                                             import copyfile
from package_utils.check_dir                            import chek_dir

def setParameters():

  p = Parameters(
    # --- relative to the training phase
    NB_EPOCH                                        = 5,                                                                                                                            # Number of training epochs
    NBPATIENCE_EPOCHS                               = 50,                                                                                                                   # Stop training if not amelioration
    BATCH_SIZE                                      = 8,                                                                                                                           # select the batch size
    DATA_AUGMENTATION                               = True,                                                                                                                 # True to apply data augmentation on training sets only
    MODEL_SELECTION                                 = 'custom_dilated_unet',                                                                                                  # The name of the desired architecture
    LEARNING_RATE                                   = 0.001,                                                                                                                    # The starting value of the learning rate
    LOSS                                            = 'dice_bce_loss',                                                                                                                   # The desired loss function
    PATH_TO_SAVE_PREDICTION_DURING_TRAINING         = '/run/media/laine/HDD/PROJECTS_IO/caroSegDeep/RESULTS/EXP0/FAR_WALL_MODEL/TRAINING/',           # Save prediction of one at the end of each epoch during training
    PATCH_HEIGHT                                    = 512,  # The height of a patch
    PATCH_WIDTH                                     = 128,  # The width of a patch

    # --- relative to results
    PATH_TO_SAVE_TENSORBOARD                        = '/run/media/laine/HDD/PROJECTS_IO/caroSegDeep/RESULTS/EXP0/FAR_WALL_MODEL/TENSORBOARD/',                           # path to save tensorboard
    PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS        = '/run/media/laine/HDD/PROJECTS_IO/caroSegDeep/RESULTS/EXP0/FAR_WALL_MODEL/PREDICTION_RESULTS/',    # where metrics/pdf are saved
    NAME_OF_THE_EXPERIMENT                          = 'EXAMPLE',                                                                                                           # name of the experiment

    # --- path to load pretrained model for fine tuning
    PATH_PRETRAINED_MODEL                           = None,  # path to lad pretrained model

    # --- relative to data
    PATH_TO_DATASET                                 = '/run/media/laine/HDD/PROJECTS_IO/caroSegDeep/DATASET/CUBS_far_wall.h5',                                                    # Path where dataset in .h5 is saved
    PATH_FOLD                                       = { \
      'training': '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/Folds/f0/TrainList.txt',                                             # Path to fold
      'validation': '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/Folds/f0/ValList.txt',
      'testing': '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/Folds/f0/TestList.txt'}
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
