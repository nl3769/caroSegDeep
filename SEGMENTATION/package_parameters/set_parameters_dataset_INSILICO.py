import os

from shutil                                             import copyfile
from package_parameters.parameters_dataset_INSILICO     import Parameters
from package_utils.check_dir                            import chek_dir

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

  p = Parameters(
    PATH_TO_SEQUENCES           = '/home/laine/Documents/PROJECTS_IO/SIMULATION/DATA',                    # Path where the sequences/images are saved (.tiff, .DICOM, .MAT)
    PATH_TO_SAVE_DATASET        = '/home/laine/Documents/PROJECTS_IO/SEGMENTATION/IN_SILICO/DATASET',     # path to save the dataset in h5 format
    PATH_TO_SKIPPED_SEQUENCES   = '/home/laine/Documents/PROJECTS_IO/SEGMENTATION/IN_SILICO/DATASET',     # If an image can't be part of the sets then it saves in .txt file
    SCALE                       = True,                                                                   # Chose true to apply data augmentation on training set only
    PATCH_WIDTH                 = 128,                                                                    # The width of a patch
    PATCH_OVERLAY               = 32,                                                                     # Number of overlapping pixels when the patch is moved
    SPATIAL_RESOLUTION          = 5,                                                                      # The desired spatial resolution by column
  )

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the package_parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  chek_dir(os.path.join(p.PATH_TO_SKIPPED_SEQUENCES, "parameters"))
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)),
           os.path.join(p.PATH_TO_SKIPPED_SEQUENCES, "parameters", 'get_parameters_training.py'))

  # --- Create dir
  chek_dir(p.PATH_TO_SAVE_DATASET)
  chek_dir(p.PATH_TO_SKIPPED_SEQUENCES)

  # --- Return populated object from Parameters class
  return p
