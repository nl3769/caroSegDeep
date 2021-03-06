class Parameters:

  def __init__(
          self,
          PATH_TO_SEQUENCES,
          PATH_TO_BORDERS,
          PATH_TO_CONTOURS,
          PATH_TO_CF,
          PROCESS_FULL_SEQUENCE,
          PATCH_HEIGHT,
          PATCH_WIDTH,
          OVERLAPPING,
          DESIRED_SPATIAL_RESOLUTION,
          PATH_WALL_SEGMENTATION_RES,
          PATH_FAR_WALL_SEGMENTATION_RES,
          PATH_MODEL_FW,
          PATH_MODEL_WALL,
          PATH_TO_LOAD_GT,
          MODEL_NAME,
          USED_FAR_WALL_DETECTION_FOR_IMC,
          PATH_TO_FOLDS
  ):

    self.PATH_TO_SEQUENCES                  = PATH_TO_SEQUENCES
    self.PATH_TO_BORDERS                    = PATH_TO_BORDERS
    self.PATH_TO_CONTOURS                   = PATH_TO_CONTOURS
    self.PATH_TO_CF                         = PATH_TO_CF
    self.PROCESS_FULL_SEQUENCE              = PROCESS_FULL_SEQUENCE
    self.PATCH_HEIGHT                       = PATCH_HEIGHT
    self.PATCH_WIDTH                        = PATCH_WIDTH
    self.OVERLAPPING                        = OVERLAPPING
    self.DESIRED_SPATIAL_RESOLUTION         = DESIRED_SPATIAL_RESOLUTION
    self.PATH_WALL_SEGMENTATION_RES         = PATH_WALL_SEGMENTATION_RES
    self.PATH_FAR_WALL_SEGMENTATION_RES     = PATH_FAR_WALL_SEGMENTATION_RES
    self.PATH_MODEL_FW                      = PATH_MODEL_FW
    self.PATH_MODEL_WALL                    = PATH_MODEL_WALL
    self.PATH_TO_LOAD_GT                    = PATH_TO_LOAD_GT
    self.MODEL_NAME                         = MODEL_NAME
    self.USED_FAR_WALL_DETECTION_FOR_IMC    = USED_FAR_WALL_DETECTION_FOR_IMC
    self.PATH_TO_FOLDS                      = PATH_TO_FOLDS