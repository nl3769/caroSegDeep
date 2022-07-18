import os
from package_parameters.parameters_inference            import Parameters
from shutil                                             import copyfile
from package_utils.check_dir                            import chek_dir

def setParameters():

  p = Parameters(
      PATH_TO_SEQUENCES                               = '/home/laine/Documents/ICCVG/SILICO-SEQUENCES/tech_004',
      PATH_TO_BORDERS                                 = '/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/BORDERS_A1',      # Path where the borders are saved (.MAT)
      PATH_TO_CONTOURS                                = '/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/CONTOURS/A2',     # Path where the contours are saved (.MAT, .txt)
      PATH_TO_CF                                      = '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/ICCVG',
      PROCESS_FULL_SEQUENCE                           = True,                                                                                                          # Segment all the frame of the sequence or only the first one
      PATCH_HEIGHT                                    = 512,                                                                                                           # Height of a patch
      PATCH_WIDTH                                     = 128,                                                                                                           # Width of a patch
      OVERLAPPING                                     = 16,                                                                                                            # Horizontal displacement of a patch
      DESIRED_SPATIAL_RESOLUTION                      = 5,                                                                                                             # Desired spatial resolution in um
      PATH_WALL_SEGMENTATION_RES                      = '/home/laine/Documents/_SEGMENTATION-TEST',                                                                # Path to save results
      PATH_FAR_WALL_SEGMENTATION_RES                  = '/home/laine/Documents/_SEGMENTATION-TEST',
      PATH_MODEL_FW                                   = '/home/laine/tux/JEAN_ZAY/RESULTS/SEGMENTATION/FW_FOLD_',                                                      # Path where the trained model is saved
      PATH_MODEL_WALL                                 = '/home/laine/tux/JEAN_ZAY/RESULTS/SEGMENTATION/IMC_FOLD_',                                                     # Path where the trained model is saved
      PATH_TO_LOAD_GT                                 = '/home/laine/cluster/PROJECTS_IO/SEGMENTATION/CONSORTIUM_MEIBURGER/CREATE_REFERENCES/RESULTS/CONTOURS',
      USED_FAR_WALL_DETECTION_FOR_IMC                 = True,                                                                                                         # If true then the predicted far wall is used to segment the IMC
      PATH_TO_FOLDS                                   = '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/Folds/f0')

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
