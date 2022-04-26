from parameters.parameters_inference import Parameters
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
# 5) The untracked file "setParameters.py" is automatically copied to the tracked file "getParameters.py" for reproducibility
# ****************************************************************

def getParameters():

  p = Parameters(PATH_TO_SEQUENCES='/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/images',                             # Path where the sequences/images are saved (.tiff, .DICOM, .MAT)
                 PATH_TO_BORDERS='/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/BORDERS_A1',     # Path where the borders are saved (.MAT)
                 PATH_TO_CONTOURS='/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/CONTOURS/A2',                 # Path where the contours are saved (.MAT, .txt)
                 PATH_TO_CF='/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/CF',
                 PROCESS_FULL_SEQUENCE=False,        # Segment all the frame of the sequence or only the first one
                 PATCH_HEIGHT=512,                   # The height of a patch
                 PATCH_WIDTH=128,                    # The width of a patch
                 OVERLAPPING=32,                      # Horizontal displacement of a patch
                 DESIRED_SPATIAL_RESOLUTION=5,       # The desired spatial resolution in um
                 PATH_WALL_SEGMENTATION_RES='/home/laine/Desktop/SEGTEST',              # Path to save results
                 PATH_FAR_WALL_SEGMENTATION_RES='/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/SEGMENTATION/FAR_WALL_CAROTID_DETECTION/RESULTS/UNION_CORRECTED/fold2/SEGMENTATION',
                 # PATH_TO_LOAD_TRAINED_MODEL_WALL='/home/laine/tux/JEAN_ZAY/RESULTS/SEGMENTATION/IMC_FOLD_2/EXAMPLE',             # Path where the trained model is saved
                 PATH_TO_LOAD_TRAINED_MODEL_WALL='/home/laine/tux/JEAN_ZAY/RESULTS/SEGMENTATION/IMC_FOLD_',             # Path where the trained model is saved
                 PATH_LO_LOAD_GT='/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/CONTOURS',
                 MODEL_NAME='custom_dilated_unet.h5',
                 USED_FAR_WALL_DETECTION_FOR_IMC=True,                                                                       # If true then the predicted far wall is used to segment the IMC
                 # PATH_TO_FOLDS='/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/Folds/f0'
                 PATH_TO_FOLDS='/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/Folds/'
                 )

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  copyfile(os.path.join('parameters', os.path.basename(__file__)), os.path.join('parameters', 'get_parameters_inference.py'))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join('parameters', 'get_parameters_inference.py'), 'rt')
  data = fid.read()
  data = data.replace('getParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join('parameters', 'get_parameters.py'), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p
