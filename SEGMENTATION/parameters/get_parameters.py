from parameters.parameters import Parameters
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

def getParameters():

  p = Parameters(

                PATH_TO_SEQUENCES='/home/nlaine/Documents/cluster/PROJECTS_IO/DATA/MEIBURGER/Images',     # Path where the sequences/images are saved (.tiff, .DICOM, .MAT)
                PATH_TO_BORDERS='/home/nlaine/Documents/cluster/PROJECTS_IO/DATA/MEIBURGER/Borders',      # Path where the borders are saved (.MAT)
                PATH_TO_CONTOURS='/home/nlaine/Documents/cluster/PROJECTS_IO/DATA/MEIBURGER/Contours',    # Path where the contours are saved (.MAT, .txt)
                PATH_TO_CF='/home/nlaine/Documents/cluster/PROJECTS_IO/DATA/MEIBURGER/CF',
                FOLD='fold1',                                                                            # The fold's name is used to load the correct model
                PATH_TO_SAVE_EDUARDO_VIEWER='/home/nlaine/Documents/cluster/projects_results/carotid_segmentation_results/results_viewer_NL/caroSeg_MEIBURGER/laine',   # Path where data are stored and where the format matches with Eduardo viewer
                PROCESS_FULL_SEQUENCE=False, # Segment all the frame of the sequence or only the first one
                PATH_TO_SAVE_RESULTS_COMPRESSION='/home/nlaine/Documents/cluster/PROJECTS_IO/CONSORTIUM_MEIBURGER/SEGMENTATION/WALL_CAROTID_SEGMENTATION/RESULTS/COMPRESSION',
                PATCH_HEIGHT=512, # The height of a patch
                PATCH_WIDTH=128,  # The width of a patch
                OVERLAPPING=8,    # Horizontal displacement of a patch
                DESIRED_SPATIAL_RESOLUTION=5, # The desired spatial resolution in um

                PATH_FAR_WALL_DETECTION_RES='/home/nlaine/Documents/cluster/PROJECTS_IO/CONSORTIUM_MEIBURGER/SEGMENTATION/FAR_WALL_CAROTID_DETECTION/RESULTS/',                  # Where the detection of the far wall was saved
                PATH_TO_LOAD_TRAINED_MODEL_FAR_WALL='/home/nlaine/Documents/cluster/PROJECTS_IO/CONSORTIUM_MEIBURGER/SEGMENTATION/FAR_WALL_CAROTID_DETECTION/DATA',              # Path where the trained model is saved

                PATH_TO_LOAD_TRAINED_MODEL_WALL='/home/nlaine/Documents/cluster/PROJECTS_IO/CONSORTIUM_MEIBURGER/SEGMENTATION/WALL_CAROTID_SEGMENTATION/DATA',               # Path where the trained model is saved
                PATH_WALL_SEGMENTATION_RES='/home/nlaine/Documents/cluster/PROJECTS_IO/CONSORTIUM_MEIBURGER/SEGMENTATION/WALL_CAROTID_SEGMENTATION/RESULTS',

                NAME_OF_TRAINED_MODEL='custom_dilated_unet.h5',                                                                                               # Name of the trained model

                FLATTEN=False,                           # flatten image in order to apply dynamic progamming

                MANUAL_FAR_WALL_DETECTION=False,          # if True then the home made GUI is used, else the prediction of the far wall are loaded
                AUTOMATIC_METHOD=False                   # run the fully automatic method
  )

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  copyfile(os.path.join('parameters', 'set_parameters.py'), os.path.join('parameters', 'get_parameters.py'))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join('parameters', 'get_parameters.py'), 'rt')
  data = fid.read()
  data = data.replace('getParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join('parameters', 'get_parameters.py'), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p
