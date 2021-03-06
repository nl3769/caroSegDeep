import os
from package_parameters.parameters_inference    import Parameters
from package_utils.check_dir                    import chek_dir
from shutil                                     import copyfile

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

def setParameters():

  p = Parameters(
      PATH_TO_SEQUENCES                              = '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/images',                                                         # Path where the sequences/images are saved (.tiff, .DICOM, .MAT)
      PATH_TO_BORDERS                                = '/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/BORDERS_A1',        # Path where the borders are saved (.MAT)
      PATH_TO_CONTOURS                               = '/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/CONTOURS/A1',       # Path where the contours are saved (.MAT, .txt)
      PATH_TO_CF                                     = '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/ICCVG/CF',
      PROCESS_FULL_SEQUENCE                          = True,                                                                                                            # Segment all the frame of the sequence or only the first one
      PATCH_HEIGHT                                   = 512,                                                                                                             # The height of a patch
      PATCH_WIDTH                                    = 128,                                                                                                             # The width of a patch
      OVERLAPPING                                    = 16,                                                                                                              # Horizontal displacement of a patch
      DESIRED_SPATIAL_RESOLUTION                     = 5,                                                                                                               # The desired spatial resolution in um
      PATH_WALL_SEGMENTATION_RES                     = '/home/laine/Desktop/ICCVG_RES',                                                                                 # Path to save results
      PATH_FAR_WALL_SEGMENTATION_RES                 = '/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/SEGMENTATION/FAR_WALL_CAROTID_DETECTION/RESULTS/UNION_CORRECTED/fold2/SEGMENTATION',
      PATH_MODEL_FW                                  = '/home/laine/tux/JEAN_ZAY/RESULTS/SEGMENTATION/IMC_FOLD_0/EXAMPLE/custom_dilated_unet.h5',                                                        # Path where the trained model is saved
      PATH_MODEL_WALL                                = '/home/laine/tux/JEAN_ZAY/RESULTS/SEGMENTATION/FW_FOLD_0/EXAMPLE/custom_dilated_unet.h5',                                                       # Path where the trained model is saved
      PATH_TO_LOAD_GT                                = '/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/CONTOURS',
      MODEL_NAME                                     = 'custom_dilated_unet.h5',
      USED_FAR_WALL_DETECTION_FOR_IMC                = True,                                                                                                           # If true then the predicted far wall is used to segment the IMC
      PATH_TO_FOLDS                                  = '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/Folds/f0')

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Make copy in results directory in order to track package_parameters
  chek_dir(os.path.join(p.PATH_WALL_SEGMENTATION_RES, "parameters"))
  copyfile(
      os.path.join('package_parameters', os.path.basename(__file__)),
      os.path.join(p.PATH_WALL_SEGMENTATION_RES, "parameters", 'get_parameters_inference.py'))

  # --- Return populated object from Parameters class
  return p
