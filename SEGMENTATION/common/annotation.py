import scipy.io

import os

from common.get_far_wall import cv2Annotation
from functions.processing import getBiggestConnexeRegion
from functions.borders_extraction import *
from numba import jit
import numpy as np



class annotationClass():

    def __init__(self,
                 dimension,
                 borderPath,
                 firstFrame,
                 scale,
                 overlay,
                 p,
                 patientName):

        self.mapAnnotation = np.zeros((dimension[0] + 1, dimension[2], 2))
        # self.fold = fold
        self.patient = patientName
        self.overlay = overlay
        self.sequenceDimension = dimension
        self.bordersROI = {}

        pos, img, bordersSeg, bordersROI = self.getFarWall(firstFrame)
        self.borders = {"leftBorder": bordersSeg[0], "rightBorder": bordersSeg[1]}
        self.bordersROI = {"leftBorder": bordersROI[0], "rightBorder": bordersROI[1]}
        self.initialization(localization=pos, scale=scale)

        # --- display borders
        keys = list(self.borders.keys())
        print(keys[0] + " = ", self.borders[keys[0]])
        print(keys[1] + " = ", self.borders[keys[1]])
    # ------------------------------------------------------------------------------------------------------------------
    def initialization(self,
                       localization,
                       scale):

        IFC3 = np.zeros(self.sequenceDimension[2])
        IFC4 = np.zeros(self.sequenceDimension[2])

        IFC3[self.bordersROI['leftBorder']:self.bordersROI['rightBorder']] = localization*scale
        IFC4[self.bordersROI['leftBorder']:self.bordersROI['rightBorder']] = localization*scale

        self.mapAnnotation[0, :, 0] = IFC3
        self.mapAnnotation[0, :, 1] = IFC4
    # ------------------------------------------------------------------------------------------------------------------
    def updateAnnotation(self,
                         previousMask,
                         frameID,
                         offset):


        # --- window of +/- neighbours pixels where the algorithm search the borders
        neighours = 30  # TODO

        # --- the algorithm starts from the left to the right
        xStart = self.bordersROI['leftBorder']
        xEnd = self.bordersROI['rightBorder']

        # --- dimension of the mask
        dim = previousMask.shape
        # --- we extract the biggest connexe region
        previousMask[previousMask > 0.5] = 1
        previousMask[previousMask < 1] = 0
        previousMask = getBiggestConnexeRegion(previousMask)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(previousMask == 1))
        seed = (round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))
        j, limit=0,dim[0]-1
        self.LI_center_to_left_propagation(j, seed, xStart, dim, previousMask, offset, self.mapAnnotation[frameID,], neighours, limit)
        j, limit=0,dim[0]-1
        self.LI_center_to_right_propagation(j, seed, xEnd, dim, previousMask, offset, self.mapAnnotation[frameID,], neighours, limit)
        j, limit = 0, dim[0]-1
        self.MA_center_to_left_propagation(j, seed, xStart, dim, previousMask, offset, self.mapAnnotation[frameID,], neighours, limit)
        j, limit = 0, dim[0]-1
        self.MA_center_to_right_propagation(j, seed, xEnd, dim, previousMask, offset, self.mapAnnotation[frameID,], neighours, limit)


        return previousMask
    # ------------------------------------------------------------------------------------------------------------------
    def updateAnnotationFlatten(self,
                                prediction,
                                frameID,
                                scale,
                                tmp):

        cumulativeCostMap, backTrackingMap = frontPropagation(costMap=np.square(prediction - 0.5) / 0.25, scale=round(scale))
        LI, MA = tracking(cumulativeCostMap=cumulativeCostMap, backTrackingMap=backTrackingMap)

        self.mapAnnotation[frameID, self.borders['leftBorder']:self.borders['rightBorder'], 0] = self.mapAnnotation[frameID-1, self.borders['leftBorder']:self.borders['rightBorder'], 0] + LI - 255
        self.mapAnnotation[frameID, self.borders['leftBorder']:self.borders['rightBorder'], 1] = self.mapAnnotation[frameID - 1, self.borders['leftBorder']:self.borders['rightBorder'], 0] + MA - 255

        for k in range(self.borders['leftBorder'], self.borders['rightBorder']+1):
            tmp[0, int(self.mapAnnotation[frameID, k, 0]), k] = 255
            tmp[0, int(self.mapAnnotation[frameID, k, 1]), k] = 255
    # ------------------------------------------------------------------------------------------------------------------
    def getFarWall(self,
                   img):
        """
        GUI with openCV with spline interpolation to localize the far wall
        """
        image = np.zeros(img.shape + (3,))
        image[:, :, 0] = img.copy()
        image[:, :, 1] = img.copy()
        image[:, :, 2] = img.copy()

        coordinateStore = cv2Annotation("Far wall manual detection", image.astype(np.uint8))

        return coordinateStore.getpt()
    # ------------------------------------------------------------------------------------------------------------------
    def IMT(self):

        '''
        Compute the IMT
        '''
        xLeft = self.borders['leftBorder']
        xRight = self.borders['rightBorder']

        IMT = self.mapAnnotation[:, xLeft:xRight, 1] - self.mapAnnotation[:, xLeft:xRight, 0]

        return np.mean(IMT, axis=1), np.median(IMT, axis=1)
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def yPosition(xLeft,
                  width,
                  height,
                  map):

        '''
        Compute the y position on which the current patch will be centered
        '''

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
    @staticmethod
    @jit(nopython=True)
    def LI_center_to_right_propagation(j, seed, xEnd, dim, previousMask, offset, mapAnnotation, neighours, limit):

        for i in range(seed[1] + 1, xEnd):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previousMask[j, i] == 1):
                    mapAnnotation[i, 0] = j + offset
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    mapAnnotation[i, 0] = mapAnnotation[i - 1, 0]
                    condition = False

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @jit(nopython=True)
    def LI_center_to_left_propagation(j, seed, xStart, dim, previousMask, offset, mapAnnotation, neighours, limit):
        for i in range(seed[1], xStart - 1, -1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previousMask[j, i] == 1):
                    mapAnnotation[i, 0] = j + offset
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    mapAnnotation[i, 0] = mapAnnotation[i + 1, 0]
                    condition = False
                    # previousMask[j, i] = 100 # for debug

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @jit(nopython=True)
    def MA_center_to_right_propagation(j, seed, xEnd, dim, previousMask, offset, mapAnnotation, neighours, limit):

        for i in range(seed[1] + 1, xEnd):
            condition = True

            while condition == True:

                if (j < dim[0] and previousMask[dim[0] - 1 - j, i] == 1):
                    mapAnnotation[i, 1] = dim[0] - 1 - j + offset
                    condition = False

                elif j == limit:
                    mapAnnotation[i, 1] = mapAnnotation[i - 1, 1]
                    condition = False
                    # previousMask[j, i] = 100

                j += 1

            j -= neighours + 1
            limit = j + 2 * neighours

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @jit(nopython=True)
    def MA_center_to_left_propagation(j, seed, xStart, dim, previousMask, offset, mapAnnotation, neighours, limit):
        for i in range(seed[1], xStart - 1, -1):
            condition = True

            while condition == True:

                if (j < dim[0] and previousMask[dim[0] - 1 - j, i] == 1):
                    mapAnnotation[i, 1] = dim[0] - 1 - j + offset
                    condition = False

                elif j == limit:
                    mapAnnotation[i, 1] = mapAnnotation[i + 1, 1]
                    condition = False

                j += 1

            j -= neighours + 1
            limit = j + 2 * neighours