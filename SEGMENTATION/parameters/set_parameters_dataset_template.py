from parameters.parameters_dataset import Parameters
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

def createDirectory(path):
  try:
    os.makedirs(path)
  except OSError:
    print("The directory %s already exists." % path)
  else:
    print("Successfully created the directory %s " % path)

  for f in os.listdir(path):
    os.remove(os.path.join(path, f))

def setParameters():

  p = Parameters(

    PATH_TO_SEQUENCES='/home/nlaine/cluster/PROJECTS_IO/DATA/MEIBURGER/images',  # Path where the sequences/images are saved (.tiff, .DICOM, .MAT)
    PATH_TO_BORDERS='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/DATASET/BORDERS/BORDERS_A1',                         # Path where the borders are saved (.mat)
    PATH_TO_CONTOUR='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/DATASET/CONTOURS/A1',                                # Path where the contours are saved (.mat, .txt)
    PATH_TO_CF='/mnt/166CCE046CCDDE9D/CUBS/CF',                             # Patch where the calibration factor is saved
    EXPERT='A1',                                                            # Name of the expert
    DATABASE_NAME = ['CUBS'],                                               # Name of the database
    PATH_TO_SAVE_DATASET='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/DATASET/DATASET',             # path to save the dataset in h5 format
    PATH_TO_SKIPPED_SEQUENCES='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/DATASET/DATASET',        # If an image can't be part of the sets then it saves in .txt file
    PATH_TO_FOLDS='/home/nlaine/Documents/REPO/caroSegDeep/EXAMPLE/RESULTS/DATASET/DATASET/DISTRIBUTION',       # In this directory .txt files contain the patient's name according to their belonging (train/val/test)
    SCALE=True,                     # Chose true to apply data augmentation on training set only
    PATCH_WIDTH=128,                # The width of a patch
    PATCH_OVERLAY=28,               # Number of overlapping pixels when the patch is moved
    SPATIAL_RESOLUTION=5,           # The desired spatial resolution by column

  )

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  copyfile(os.path.join('parameters', os.path.basename(__file__)), os.path.join('parameters', 'get_parameters_dataset.py'))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join('parameters', 'get_parameters_dataset.py'), 'rt')
  data = fid.read()
  data = data.replace('setParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join('parameters', 'get_parameters_dataset.py'), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p