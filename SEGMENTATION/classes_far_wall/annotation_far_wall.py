import scipy.io

import os

from common.get_far_wall import cv2Annotation
from functions.processing import getBiggestConnexeRegion
from functions.bordersExtraction import *


class annotationClass():

    def __init__(self,
                 dimension,
                 borderPath,
                 firstFrame,
                 scale,
                 fold,
                 overlay,
                 p,
                 patientName=None):

        self.mapAnnotation = np.zeros((dimension[0] + 1, dimension[2], 2))
        self.fold = fold
        self.patient = patientName
        self.overlay = overlay
        self.sequenceDimension = dimension

        if p.MANUAL_FAR_WALL_DETECTION == True and p.AUTOMATIC_METHOD==False:
            pos, img, bordersROI, bordersSeg = self.getFarWall(firstFrame)
            self.borders = {"leftBorder": bordersSeg[0], "rightBorder": bordersSeg[1]}
            self.bordersROI = {"leftBorder": bordersROI[0], "rightBorder": bordersROI[1]}

            self.annotation = self.initialization(localization=pos,
                                                  scale=scale)

        elif p.MANUAL_FAR_WALL_DETECTION == False and p.AUTOMATIC_METHOD==False:

            self.bordersROI = self.LoadBorderMeiburger(borderPath)
            # self.borders = {"leftBorder": self.borders["leftBorder"]+overlay-128,
            #                 "rightBorder": self.borders["rightBorder"]-overlay+128}
            self.borders = self.bordersROI.copy()
            # self.LoadPredictedAnnotationLinearHypothesis(scale=scale)
            self.LoadPredictedAnnotation(scale=scale, p=p)

        elif p.AUTOMATIC_METHOD==True:
            self.borders = self.LoadBorders(borderPath)
            self.bordersROI = self.borders
            self.mapAnnotation = np.zeros((dimension[0] + 1, dimension[2], 2))
            self.sequenceDimension = dimension


        #### Remove later ####
        keys = list(self.borders.keys())
        print(keys[0] + " = ", self.borders[keys[0]])
        print(keys[1] + " = ", self.borders[keys[1]])

    # ------------------------------------------------------------------------------------------------------------------
    def initialization(self,
                       localization,
                       scale):

        IFC3 = np.zeros(self.sequenceDimension[2])
        IFC4 = np.zeros(self.sequenceDimension[2])

        IFC3[self.borders['leftBorder']:self.borders['rightBorder']] = localization*scale
        IFC4[self.borders['leftBorder']:self.borders['rightBorder']] = localization*scale

        self.mapAnnotation[0, :, 0] = IFC3
        self.mapAnnotation[0, :, 1] = IFC4

        return {"IFC3": self.mapAnnotation[0, :, 0], "IFC4": self.mapAnnotation[0, :, 1]}

    # ------------------------------------------------------------------------------------------------------------------
    def updateAnnotation(self,
                         previousMask,
                         frameID):


        # --- window of +/- neighbours pixels where the algorithm search the borders
        neighours = 30

        # --- the algorithm starts from the left to the right
        xStart = self.borders['leftBorder']
        xEnd = self.borders['rightBorder']

        # --- dimension of the mask
        dim = previousMask.shape
        # --- we extract the biggest connexe region
        previousMask[previousMask > 0.5] = 1
        previousMask[previousMask < 1] = 0
        previousMask = getBiggestConnexeRegion(previousMask)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(previousMask == 1))
        seed = (round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))
        # --- random value to j but not to high. It the first y coordinate at the xStart position
        j = 300
        # --- j cannot exceed limit
        limit = dim[0] - 1
        # --- delimitation of the LI boundary
        for i in range(seed[1] + 1, xEnd + 1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previousMask[j, i] == 1):
                    self.mapAnnotation[frameID, i, 0] = j
                    if self.mapAnnotation[frameID, i, 0] == 0:
                        print(2)
                    # previousMask[j, i] = 100 # for debug
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    self.mapAnnotation[frameID, i, 0] = self.mapAnnotation[frameID, i - 1, 0]
                    condition = False
                    if self.mapAnnotation[frameID, i, 0] == 0:
                        print(2)
                    # previousMask[j, i] = 100 # for debug

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours
        j = 300
        limit = dim[0] - 1
        for i in range(seed[1], xStart - 1, -1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previousMask[j, i] == 1):
                    self.mapAnnotation[frameID, i, 0] = j
                    # previousMask[j, i] = 100 # for debug
                    if self.mapAnnotation[frameID, i, 0] == 0:
                        print(2)
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    self.mapAnnotation[frameID, i, 0] = self.mapAnnotation[frameID, i + 1, 0]
                    if self.mapAnnotation[frameID, i, 0] == 0:
                        print(2)
                    condition = False
                    # previousMask[j, i] = 100 # for debug

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours

        # --- delimitation of the MA boundary
        j = 300
        limit = dim[0] - 1
        for i in range(seed[1] + 1, xEnd + 1):
            condition = True

            while condition == True:

                if (j < dim[0] and previousMask[dim[0] - 1 - j, i] == 1):
                    self.mapAnnotation[frameID, i, 1] = dim[0] - 1 - j
                    # previousMask[dim[0] - 1 - j, i] = 100
                    if self.mapAnnotation[frameID, i, 0] == 0:
                        print(2)
                    condition = False

                elif j == limit:
                    self.mapAnnotation[frameID, i, 1] = self.mapAnnotation[frameID, i - 1, 1]
                    if self.mapAnnotation[frameID, i, 0] == 0:
                        print(2)
                    condition = False
                    # previousMask[j, i] = 100

                j += 1

            j -= neighours + 1
            limit = j + 2 * neighours

        j = 300
        limit = dim[0] - 1
        for i in range(seed[1], xStart - 1, -1):
            condition = True

            while condition == True:

                if (j < dim[0] and previousMask[dim[0] - 1 - j, i] == 1):
                    self.mapAnnotation[frameID, i, 1] = dim[0] - 1 - j
                    if i == 315:
                        print(2)
                    # previousMask[dim[0] - 1 - j, i] = 100
                    condition = False

                elif j == limit:
                    self.mapAnnotation[frameID, i, 1] = self.mapAnnotation[frameID, i + 1, 1]
                    if i == 315:
                        print(2)
                    condition = False
                    # previousMask[j, i] = 100

                j += 1

            j -= neighours + 1
            limit = j + 2 * neighours

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

        print(2)

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

        coordinateStore = cv2Annotation("First frame of the sequence", image.astype(np.uint8))

        return coordinateStore.getpt()

    # ------------------------------------------------------------------------------------------------------------------
    def LoadAnnotation(self,
                       contours_path,
                       nameAnnotationIFC3,
                       nameAnnotationIFC4,
                       scale):
        """
        Locate the far wall using the experts' annotation.
        """
        mat_IFC3 = scipy.io.loadmat(contours_path + nameAnnotationIFC3)
        mat_IFC4 = scipy.io.loadmat(contours_path + nameAnnotationIFC4)

        self.mapAnnotation[0, :, 0] = scale*(mat_IFC3['seg'][:, 0])
        self.mapAnnotation[0, :, 1] = scale*(mat_IFC4['seg'][:, 0])

        # we remove the annotation which are out of the borders
        self.mapAnnotation[0, 0:self.borders['leftBorder']+1, 0] = 0
        self.mapAnnotation[0, self.borders['rightBorder']:, 0] = 0
        self.mapAnnotation[0, 0:self.borders['leftBorder']+1, 1] = 0
        self.mapAnnotation[0, self.borders['rightBorder']: , 1] = 0

        return {"IFC3": self.mapAnnotation[0, :, 0], "IFC4": self.mapAnnotation[0, :, 1]}

    # ------------------------------------------------------------------------------------------------------------------
    def IMT(self):

        '''
        Compute the IMT
        '''
        xLeft = self.bordersROI['leftBorder']
        xRight = self.bordersROI['rightBorder']

        IMT = self.mapAnnotation[:, xLeft:xRight, 1] - self.mapAnnotation[:, xLeft:xRight, 0]

        return np.mean(IMT, axis=1), np.median(IMT, axis=1)

    # ------------------------------------------------------------------------------------------------------------------
    def LoadPredictedAnnotationLinearHypothesis(self,
                                                scale):
        '''
        Load the far wall prediction to segment the boundaries, and a linear extrapolation is applied on the boundaries to extend the ROI and avoid boundary problems.
        '''
        pathToPredictionFW = "../../far_wall_carotide_detection/results/" + self.fold + "/segmentation/"
        mat_IFC3 = self.loadPrediction(self.patient.split('.')[0], pathToPredictionFW)
        mat_IFC4 = mat_IFC3.copy()

        self.mapAnnotation[0, :, 0] = scale*mat_IFC3
        self.mapAnnotation[0, :, 1] = scale*mat_IFC4

        # we remove the annotation which are out of the borders
        self.mapAnnotation[0, 0:self.borders['leftBorder'], 0] = 0
        self.mapAnnotation[0, self.borders['rightBorder']:, 0] = 0
        self.mapAnnotation[0, 0:self.borders['leftBorder'], 1] = 0
        self.mapAnnotation[0, self.borders['rightBorder']:, 1] = 0

        # --- we modidy the array using linear interpolation
        # left border
        x_a = self.borders['leftBorder']
        y_a = self.mapAnnotation[0, x_a, 0]

        x_b = self.borders['leftBorder']+128
        y_b = self.mapAnnotation[0, x_b, 0]

        slop_L = (y_b-y_a)/(x_b-x_a)

        if  self.borders['leftBorder']+self.overlay-128 > 0:
            self.borders['leftBorder'] = self.borders['leftBorder'] + self.overlay - 128
            limit = self.borders['leftBorder']
        else:
            self.borders['leftBorder'] = 0
            limit = 0

        for k in range(limit, self.bordersROI['leftBorder'], 1):
            if slop_L*(k-x_a) + y_a + 150 < self.sequenceDimension[1] and slop_L*(k-x_a) + y_a >150:
                self.mapAnnotation[0, k, 0] = slop_L*(k-x_a) + y_a
            elif slop_L*(k-x_a) + y_a <= 128:
                self.mapAnnotation[0, k, 0] = 150
            elif slop_L*(k-x_a) + y_a + 150 >= self.sequenceDimension[1]:
                self.mapAnnotation[0, k, 0] = self.sequenceDimension[1]-150

        # right border
        x_a = self.borders['rightBorder']-1
        y_a = self.mapAnnotation[0, x_a, 0]

        x_b = self.borders['rightBorder']-128
        y_b = self.mapAnnotation[0, x_b, 0]

        slop_R = (y_b-y_a)/(x_b-x_a)

        if self.borders['rightBorder']-self.overlay+128 < self.sequenceDimension[2]:
            self.borders['rightBorder'] = self.borders['rightBorder'] - self.overlay + 128
            limit = self.borders['rightBorder']
        else:
            self.borders['rightBorder'] = self.sequenceDimension[2] - 1
            limit = self.borders['rightBorder']


        for k in range(self.bordersROI['rightBorder'], limit, 1):
            if slop_R*(k-x_a) + y_a + 150 < self.sequenceDimension[1] and slop_R*(k-x_a) + y_a > 150:
                self.mapAnnotation[0, k, 0] = slop_R*(k-x_a) + y_a
            elif slop_R*(k-x_a) + y_a <= 128:
                self.mapAnnotation[0, k, 0] = 150
            elif slop_R * (k - x_a) + y_a + 150 >= self.sequenceDimension[1]:
                self.mapAnnotation[0, k, 0] = self.sequenceDimension[1]-150


        self.mapAnnotation[0, :, 1] = self.mapAnnotation[0, :, 0]

        return {"IFC3": self.mapAnnotation[0, :, 0], "IFC4": self.mapAnnotation[0, :, 1]}

    # ------------------------------------------------------------------------------------------------------------------
    def LoadPredictedAnnotation(self,
                                scale,
                                p):
        '''
        Load the far wall prediction to segment the borders
        '''
        pathToPredictionFW = os.path.join(p.PATH_FAR_WALL_DETECTION, self.fold + '/SEGMENTATION/')

        mat_IFC3 = self.loadPrediction(self.patient.split('.')[0], pathToPredictionFW)
        mat_IFC4 = mat_IFC3.copy()

        self.mapAnnotation[0, :, 0] = scale*mat_IFC3
        self.mapAnnotation[0, :, 1] = scale*mat_IFC4

        # # we remove the annotation which are out of the borders
        # self.mapAnnotation[0, 0:self.borders['leftBorder'], 0] = 0
        # self.mapAnnotation[0, self.borders['rightBorder']:, 0] = 0
        # self.mapAnnotation[0, 0:self.borders['leftBorder'], 1] = 0
        # self.mapAnnotation[0, self.borders['rightBorder']:, 1] = 0


        self.mapAnnotation[0, :, 1] = self.mapAnnotation[0, :, 0]

        # --- check borders
        tmpLI = np.where(self.mapAnnotation[0, :, 0] != 0)[0]
        self.borders['rightBorder'] = tmpLI.max()
        self.borders['leftBorder'] = tmpLI.min()
        self.bordersROI['rightBorder'] = tmpLI.max()
        self.bordersROI['leftBorder'] = tmpLI.min()
        print(2)

        return {"IFC3": self.mapAnnotation[0, :, 0], "IFC4": self.mapAnnotation[0, :, 1]}

    # ------------------------------------------------------------------------------------------------------------------
    def FWAutoInitialization(self, img, seed):

        # --- window of +/- neighbours pixels where the algorithm search the borders
        neighours = 10

        # --- the algorithm starts from the left to the right
        xStart = self.borders['leftBorder']
        xEnd = self.borders['rightBorder']

        # --- dimension of the mask
        dim = img.shape

        # --- random value to j but not to high. It the first y coordinate at the xStart position
        j = 5

        # --- j cannot exceed limit
        limit = dim[0] - 1
        # --- delimitation of the LI boundary
        for i in range(seed[1], xEnd + 1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and img[j, i] == 1):
                    self.mapAnnotation[0, i, 0] = j
                    self.mapAnnotation[0, i, 1] = j
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    self.mapAnnotation[0, i, 0] = self.mapAnnotation[0, i - 1, 0]
                    self.mapAnnotation[0, i, 1] = self.mapAnnotation[0, i - 1, 1]
                    condition = False

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours

        j = 5
        for i in range(seed[1], xStart - 1, -1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and img[j, i] == 1):
                    self.mapAnnotation[0, i, 0] = j
                    self.mapAnnotation[0, i, 1] = j
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    self.mapAnnotation[0, i, 0] = self.mapAnnotation[0, i - 1, 0]
                    self.mapAnnotation[0, i, 1] = self.mapAnnotation[0, i - 1, 1]
                    condition = False

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def LoadBorderMeiburger(borders_path):

        '''
        Load the border which contains the highest contrast
        '''

        mat_b = scipy.io.loadmat(borders_path)
        if 'border_right' in mat_b.keys():
            right_b = mat_b['border_right']
        else:
            right_b = mat_b['rightBorder']

        right_b = right_b[0, 0] - 1

        if 'border_left' in mat_b.keys():
            left_b = mat_b['border_left']
        else:
            left_b = mat_b['leftBorder']

        left_b = left_b[0, 0] - 1

        return {"leftBorder": left_b,
                "rightBorder": right_b}

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def loadPrediction( patientName,
                        path):
        '''
        Load the far wall prediction to segment the borders
        '''
        predN = open(os.path.join(path, patientName), "r")
        prediction = predN.readlines()
        predN.close()

        pred = np.zeros(len(prediction))

        for k in range(len(prediction)):
            pred[k] = prediction[k].split('\n')[0].split(' ')[-1]

        return pred

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def LoadBorders(borders_path):
        '''
        Load the right and left border
        '''
        mat_b = scipy.io.loadmat(borders_path)
        right_b = mat_b['border_right']
        right_b = right_b[0, 0] - 1
        left_b = mat_b['border_left']
        left_b = left_b[0, 0] - 1

        return {"leftBorder": left_b,
                "rightBorder": right_b}

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
        y_min = round(np.max(concatenation) - height / 2)

        # --- we check if the value is greater than zero to avoid problems
        maxheight = map.shape[1] - 1
        if (y_mean < 0 and y_mean > maxheight) or (y_min < 0) or (y_max < 0):
            print("Problem with yPosition !!!!!!")
            y_mean, y_min, y_max = 0, 0, 0

        return y_mean, y_min, y_max