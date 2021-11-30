'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from classes.cv2Annotation import cv2Annotation
from functions.get_biggest_connexe_region import get_biggest_connexe_region
from numba import jit
import numpy as np

class annotationClass():

    ''' annotationClass contains functions to:
        - update annotations
        - initialize the annotation maps
        - compute the intima-media thickness '''

    def __init__(self, dimension: tuple, first_frame: np.ndarray, scale: float, overlay: int, patient_name: str):

        self.map_annotation = np.zeros((dimension[0] + 1, dimension[2], 2))
        self.patient = patient_name
        self.overlay = overlay
        self.seq_dimension = dimension
        self.borders_ROI = {}
        pos, _, borders_seg, borders_ROI = self.get_far_wall(first_frame)
        self.borders = {"leftBorder": borders_seg[0], "rightBorder": borders_seg[1]}
        self.borders_ROI = {"leftBorder": borders_ROI[0], "rightBorder": borders_ROI[1]}
        self.initialization(localization=pos, scale=scale)

        # --- display borders
        keys = list(self.borders.keys())
        print(keys[0] + " = ", self.borders[keys[0]])
        print(keys[1] + " = ", self.borders[keys[1]])
    # ------------------------------------------------------------------------------------------------------------------
    def initialization(self, localization: np.ndarray, scale: float):
        ''' Initialize map_annotation with the manual delineation. '''
        IFC3 = np.zeros(self.seq_dimension[2])
        IFC4 = np.zeros(self.seq_dimension[2])

        IFC3[self.borders_ROI['leftBorder']:self.borders_ROI['rightBorder']] = localization*scale
        IFC4[self.borders_ROI['leftBorder']:self.borders_ROI['rightBorder']] = localization*scale

        self.map_annotation[0, :, 0] = IFC3
        self.map_annotation[0, :, 1] = IFC4
    # ------------------------------------------------------------------------------------------------------------------
    def update_annotation(self, previous_mask: np.ndarray, frame_ID: int, offset: int):
        ''' Computes the position of the LI and MA interfaces according to the predicted mask. '''
        # --- window of +/- neighbours pixels where the algorithm search the borders
        neighours = 30  
        # --- the algorithm starts from the left to the right
        x_start = self.borders_ROI['leftBorder']
        x_end = self.borders_ROI['rightBorder']
        # --- dimension of the mask
        dim = previous_mask.shape
        # --- we extract the biggest connexe region
        previous_mask[previous_mask > 0.5] = 1
        previous_mask[previous_mask < 1] = 0
        previous_mask = get_biggest_connexe_region(previous_mask)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(previous_mask == 1))
        seed = (round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))
        j, limit=0,dim[0]-1
        self.LI_center_to_left_propagation(j, seed, x_start, dim, previous_mask, offset, self.map_annotation[frame_ID,], neighours, limit)
        j, limit=0,dim[0]-1
        self.LI_center_to_right_propagation(j, seed, x_end, dim, previous_mask, offset, self.map_annotation[frame_ID,], neighours, limit)
        j, limit = 0, dim[0]-1
        self.MA_center_to_left_propagation(j, seed, x_start, dim, previous_mask, offset, self.map_annotation[frame_ID,], neighours, limit)
        j, limit = 0, dim[0]-1
        self.MA_center_to_right_propagation(j, seed, x_end, dim, previous_mask, offset, self.map_annotation[frame_ID,], neighours, limit)

        return previous_mask
    # ------------------------------------------------------------------------------------------------------------------
    def get_far_wall(self, img: np.ndarray):
        """ GUI with openCV with spline interpolation to localize the far wall. """
        image = np.zeros(img.shape + (3,))
        image[:, :, 0] = img.copy()
        image[:, :, 1] = img.copy()
        image[:, :, 2] = img.copy()

        coordinateStore = cv2Annotation("Far wall manual detection", image.astype(np.uint8))

        return coordinateStore.getpt()
    # ------------------------------------------------------------------------------------------------------------------
    def IMT(self):
        ''' Compute the IMT. '''
        xLeft = self.borders['leftBorder']
        xRight = self.borders['rightBorder']

        IMT = self.map_annotation[:, xLeft:xRight, 1] - self.map_annotation[:, xLeft:xRight, 0]

        return np.mean(IMT, axis=1), np.median(IMT, axis=1)
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def yPosition(xLeft: int, width: int, height: int, map: p.ndarray):
        ''' Compute the y position on which the current patch will be centered. '''
        xRight = xLeft + width

        # --- we load the position of the LI and MA interfaces
        posLI = map[:, 0][xLeft:xRight]
        posMA = map[:, 1][xLeft:xRight]

        # --- we compute the mean value and retrieve the half height of a patch
        concatenation = np.concatenate((posMA, posLI))
        y_mean = round(np.mean(concatenation) - height / 2)
        y_max = round(np.max(concatenation) - height / 2)
        y_min = round(np.min(concatenation) - height / 2)

        # --- we check if the value is greater than zero to avoid problems
        maxheight = map.shape[1] - 1
        if (y_mean < 0 and y_mean > maxheight) or (y_min < 0) or (y_max < 0):
            print("Problem with yPosition !!!!!!")
            y_mean, y_min, y_max = 0, 0, 0

        return y_mean, y_min, y_max
    # ------------------------------------------------------------------------------------------------------------------
    # --- use numba to be faster
    @staticmethod
    @jit(nopython=True)
    def LI_center_to_right_propagation(j: int, seed: tuple, x_end: int, dim: tuple, previous_mask: np.ndarray, offset: int, map_annotation: np.ndarray, neighours: int, limit: int):
        ''' Computes the LI interface from the center to the right. '''
        for i in range(seed[1] + 1, x_end):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previous_mask[j, i] == 1):
                    map_annotation[i, 0] = j + offset
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    map_annotation[i, 0] = map_annotation[i - 1, 0]
                    condition = False

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours
    # ------------------------------------------------------------------------------------------------------------------
    # --- use numba to be faster
    @staticmethod
    @jit(nopython=True)
    def LI_center_to_left_propagation(j: int, seed: tuple, x_start: int, dim: tuple, previous_mask: np.ndarray, offset: int, map_annotation: np.ndarray, neighours: int, limit: int):
        ''' Computes the LI interface from the center to the left. '''
        for i in range(seed[1], x_start - 1, -1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previous_mask[j, i] == 1):
                    map_annotation[i, 0] = j + offset
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    map_annotation[i, 0] = map_annotation[i + 1, 0]
                    condition = False
                    # previous_mask[j, i] = 100 # for debug

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours
    # ------------------------------------------------------------------------------------------------------------------
    # --- use numba to be faster
    @staticmethod
    @jit(nopython=True)
    def MA_center_to_right_propagation(j: int, seed: tuple, x_end: int, dim: tuple, previous_mask: np.ndarray, offset: int, map_annotation: np.ndarray, neighours: int, limit: int):
        ''' Computes the MA interface from the center to the right. '''
        for i in range(seed[1] + 1, x_end):
            condition = True

            while condition == True:

                if (j < dim[0] and previous_mask[dim[0] - 1 - j, i] == 1):
                    map_annotation[i, 1] = dim[0] - 1 - j + offset
                    condition = False

                elif j == limit:
                    map_annotation[i, 1] = map_annotation[i - 1, 1]
                    condition = False
                    # previous_mask[j, i] = 100

                j += 1

            j -= neighours + 1
            limit = j + 2 * neighours
    # ------------------------------------------------------------------------------------------------------------------
    # --- use numba to be faster
    @staticmethod
    @jit(nopython=True)
    def MA_center_to_left_propagation(j: int, seed: tuple, x_start: int, dim: tuple, previous_mask: np.ndarray, offset: int, map_annotation: np.ndarray, neighours: int, limit: int):
        ''' Computes the MA interface from the center to the left. '''
        for i in range(seed[1], x_start - 1, -1):
            condition = True

            while condition == True:

                if (j < dim[0] and previous_mask[dim[0] - 1 - j, i] == 1):
                    map_annotation[i, 1] = dim[0] - 1 - j + offset
                    condition = False

                elif j == limit:
                    map_annotation[i, 1] = map_annotation[i + 1, 1]
                    condition = False

                j += 1

            j -= neighours + 1
            limit = j + 2 * neighours