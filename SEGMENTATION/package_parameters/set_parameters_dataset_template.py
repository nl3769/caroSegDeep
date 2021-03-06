import os
from shutil import copyfile

from package_parameters.parameters_dataset              import Parameters
from package_utils.check_dir                            import chek_dir

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
    PATH_TO_SEQUENCES                 =  '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/images',                            # Path where the sequences/images are saved (.tiff, .DICOM, .MAT)
    PATH_TO_BORDERS                   =  '/run/media/laine/HDD/PROJECTS_IO/caroSegDeep/REFERENCES/BORDERS/BORDERS_A1',       # Path where the borders are saved (.mat)
    PATH_TO_CONTOUR                   =  '/run/media/laine/HDD/PROJECTS_IO/caroSegDeep/REFERENCES/CONTOURS/A1',              # Path where the contours are saved (.mat, .txt)
    PATH_TO_CF                        =  '/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/CF',                                # Patch where the calibration factor is saved
    EXPERT                            =  'A1',                                                                               # Name of the expert
    DATABASE_NAME                     =  ['CUBS'],                                                                           # Name of the database
    PATH_TO_SAVE_DATASET              =  '/home/laine/Desktop/_datasetTest',                                                 # path to save the dataset in h5 format
    PATH_TO_SKIPPED_SEQUENCES         =  '/home/laine/Desktop/_datasetTest',                                                 # If an image can't be part of the sets then it saves in .txt file
    PATH_TO_FOLDS                     =  '/run/media/laine/HDD/PROJECTS_IO/caroSegDeep/DATASET',                             # In this directory .txt files contain the patient's name according to their belonging (train/val/test)
    SCALE                             =  True,                                                                               # Chose true to apply data augmentation on training set only
    PATCH_WIDTH                       =  128,                                                                                # The width of a patch
    PATCH_OVERLAY                     =  28,                                                                                 # Number of overlapping pixels when the patch is moved
    SPATIAL_RESOLUTION                =  5,                                                                                  # The desired spatial resolution by column
  )

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Create dir
  chek_dir(os.path.join(p.PATH_TO_SKIPPED_SEQUENCES, "parameters"))
  chek_dir(p.PATH_TO_SAVE_DATASET)
  chek_dir(p.PATH_TO_SKIPPED_SEQUENCES)

  # --- Save a backup of the package_parameters
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)),
           os.path.join(p.PATH_TO_SKIPPED_SEQUENCES, "parameters", 'get_parameters_training.py'))


  # --- Return populated object from Parameters class
  return p
