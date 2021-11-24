from parameters.parameters_training import Parameters
import os
from shutil import copyfile

# ****************************************************************
# *** HOWTO
# ****************************************************************
# 0) Do not modify this template file "setParameterstemplate.py"
# 1) Create a new copy of this file "setParametersTemplate.py" and rename it into "setParameters.py"
# 2) Indicate all the variables according to your local environment and experiment
# 3) Use your own "setParameters.py" file to run the code
# 4) Do not commit/push your own "setParameters.py" file to the collective repository, it is not relevant for other people
# 5) The untracked file "setParameters.py" is automatically copied to the tracked file "getParameters.py" for reproductibility
# ****************************************************************

def create_directory(path):
  try:
    os.makedirs(path)
  except OSError:
    print("The directory %s already exists." % path)
  else:
    print("Successfully created the directory %s " % path)

  for f in os.listdir(path):
    os.remove(os.path.join(path, f))

def getParameters():

  p = Parameters(
    # --- relative to the training phase
    NB_EPOCH=2,                                 # Number of training epochs
    NBPATIENCE_EPOCHS=30,                        # The training is stopped if the loss on validation data does not improve after NBPATIENCE_EPOCHS epochs
    BATCH_SIZE=2,                                # select the batch size
    DATA_AUGMENTATION=True,                       # True to apply data augmentation on training sets only
    MODEL_SELECTION='custom_dilated_unet',        # The name of the desired architecture
    LEARNING_RATE=0.001,                          # The starting value of the learning rate
    LOSS='dice_bce_loss',                         # The desired loss function
    PATH_TO_SAVE_PREDICTION_DURING_TRAINING='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/TRAINING', # An image is predicted at the end of an epoch during training, and will be saved in the specified path.
    PATCH_HEIGHT=512,  # The height of a patch
    PATCH_WIDTH=128,  # The width of a patch

    # --- relative to results
    PATH_TO_SAVE_TENSORBOARD='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/TRAINING/TENSORBOARD/',                           # path to save tensorboard
    PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/TRAINING/PREDICTION_RESULTS/',    # where metrics/pdf are saved
    NAME_OF_THE_EXPERIMENT='EXAMPLE', # name of the experiment will appear in the directories where the results are stored. !!!!!!!!! FOLD !!!!!!!!!

    # --- relative to data
    PATH_TO_DATASET='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/DATASET/DATASET/CUBS.h5', # Path where dataset in .h5 is saved
    )

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Make copy in results directory in order to track parameters
  create_directory(os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT))
  copyfile(os.path.join('parameters', os.path.basename(__file__)),
           os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, 'get_parameters_training.py'))  # we also copy the retained parameters in the results
  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, 'get_parameters_training.py'), 'rt')
  data = fid.read()
  data = data.replace('getParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join(p.PATH_TO_SAVE_RESULTS_PDF_METRICS_WEIGHTS, p.NAME_OF_THE_EXPERIMENT, 'get_parameters_training.py'),
             'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p
