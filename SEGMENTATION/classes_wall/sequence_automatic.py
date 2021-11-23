import scipy.io

from common.getFarWall import cv2Annotation
from functions.bordersExtraction import *


class annotationClass():

    def __init__(self,
                 annotationPath,
                 nameAnnotation,
                 dimension,
                 borderPath,
                 firstFrame,
                 scale,
                 fold,
                 overlay,
                 p,
                 semiAutomatic=False,
                 patientName=None):


        # self.borders = LoadBorders(borderPath)

        self.mapAnnotation = np.zeros((dimension[0] + 1, dimension[2], 2))
        self.fold = fold
        self.patient = patientName
        self.overlay = overlay
        self.sequenceDimension = dimension
        if p.MANUAL_FAR_WALL_DETECTION == True:
            pos, img, bordersROI, bordersSeg = self.getFarWall(firstFrame)
            self.borders = {"leftBorder": bordersSeg[0], "rightBorder": bordersSeg[1]}
            self.bordersROI = {"leftBorder": bordersROI[0], "rightBorder": bordersROI[1]}

            self.annotation = self.initialization(contours_path=annotationPath,
                                                  localization=pos,
                                                  scale=scale)
        else:

            self.bordersROI = self.LoadBorderMeiburger(borderPath)
            # self.borders = {"leftBorder": self.borders["leftBorder"]+overlay-128,
            #                 "rightBorder": self.borders["rightBorder"]-overlay+128}
            self.borders = self.bordersROI.copy()
            # self.LoadPredictedAnnotationLinearHypothesis(scale=scale)
            self.LoadPredictedAnnotation(scale=scale, p=p)



        #### Remove later ####
        keys = list(self.borders.keys())
        print(keys[0] + " = ", self.borders[keys[0]])
        print(keys[1] + " = ", self.borders[keys[1]])

    @staticmethod
    def yPosition(xLeft,
                  width,
                  height,
                  map):

        xRight = xLeft + width

        # --- we load the position of the LI and MA interfaces
        posLI = map[:, 0][xLeft:xRight]
        posMA = map[:, 1][xLeft:xRight]

        # --- we compute the mean value and retrieve the half height of a patch
        concatenation = np.concatenate((posMA, posLI))
        y_mean = round(np.mean(concatenation) - height/2)
        y_max =  round(np.max(concatenation) - height/2)
        y_min = round(np.max(concatenation) - height / 2)

        # --- we check if the value is greater than zero to avoid any problem
        maxheight = map.shape[1] - 1
        if (y_mean < 0  and y_mean > maxheight) or (y_min < 0) or (y_max < 0):
            print("Problem in function yPosition !!!!!!")
            y_mean, y_min, y_max = 0, 0, 0


        return y_mean, y_min, y_max


    def initialization(self,
                       contours_path,
                       localization,
                       scale):

        IFC3 = np.zeros(self.sequenceDimension[2])
        IFC4 = np.zeros(self.sequenceDimension[2])

        IFC3[self.borders['leftBorder']:self.borders['rightBorder']] = localization*scale
        IFC4[self.borders['leftBorder']:self.borders['rightBorder']] = localization*scale

        self.mapAnnotation[0, :, 0] = IFC3
        self.mapAnnotation[0, :, 1] = IFC4

        return {"IFC3": self.mapAnnotation[0, :, 0], "IFC4": self.mapAnnotation[0, :, 1]}

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

    def getFarWall(self, img):
        """
        GUI with openCV with spline interpolation to localize the far wall
        """
        image = np.zeros(img.shape + (3,))
        image[:, :, 0] = img.copy()
        image[:, :, 1] = img.copy()
        image[:, :, 2] = img.copy()

        coordinateStore = cv2Annotation("First frame of the sequence", image.astype(np.uint8))

        return coordinateStore.getpt()

    def LoadAnnotation(self,
                       contours_path,
                       nameAnnotationIFC3,
                       nameAnnotationIFC4,
                       scale):
        """
        to position the far wall using experts' annotation
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

    def IMT(self):

        xLeft = self.bordersROI['leftBorder']
        xRight = self.bordersROI['rightBorder']

        IMT = self.mapAnnotation[:, xLeft:xRight, 1] - self.mapAnnotation[:, xLeft:xRight, 0]

        return np.mean(IMT, axis=1), np.median(IMT, axis=1)

    def LoadPredictedAnnotationLinearHypothesis(self, scale):

        pathToPredictionFW = "../../far_wall_carotide_detection/results/" + self.fold + "/segmentation/"
        mat_IFC3 = loadPrediction(self.patient.split('.')[0], pathToPredictionFW)
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

    def LoadPredictedAnnotation(self, scale, p):

        pathToPredictionFW = os.path.join(p.PATH_FAR_WALL_DETECTION, self.fold + '/SEGMENTATION/')

        mat_IFC3 = loadPrediction(self.patient.split('.')[0], pathToPredictionFW)
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

    def LoadBorderMeiburger(self, borders_path):



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


def loadPrediction(patientName, path):

    predN = open(os.path.join(path, patientName), "r")
    prediction = predN.readlines()
    predN.close()

    pred = np.zeros(len(prediction))

    for k in range(len(prediction)):
        pred[k] = prediction[k].split('\n')[0].split(' ')[-1]

    return pred
from classes_wall.prediction import predictionClass
# from classes.annotation import annotationClassTmp
from common.annotationAutomatic import annotationClass
from functions.LoadDatas import LoadData
from functions.processing import getBiggestConnexeRegion
from functions.saveResults import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

class sequenceClass():

    def __init__(self,
                 sequencePath,
                 pathAnnotation,
                 names,
                 pathBorders,
                 fullSq=False):

        '''

        ----------------- Parameters -----------------

        param: sequencePath: path to load the sequence
        param: sequencePath: path to load the annotation.   ##Todo: It will the removed later
        param: names: name of the patient.                  ##Todo: It will the removed later
        param: pathBorders:                                 ##Todo: It will the removed later
        param: fullSq: if the fullSq=True then all the image of the sequence is segmented and only the first frame if fullSq=False. By default, fullSq=False


        ----------------- Attributes -----------------

        param: self.desiredYSpacing: the desired spatial resolution in um
        param: self.fullSq: if we segment the first frame or the entire sequence (False or True)
        param: self.patchHeight: height of the patch
        param: self.pathWidth: width of the patch
        param: self.overlay: defined how much pixel does the patch is moved to the right
        param: self.sequence: rescaled sequence to reach a vertical spacing of self.desiredYSpacing
        param: self.scale: scale coefficient to reach a vertical spacing of self.desiredYSpacing
        param: self.spacing:  the original spacing of the sequence
        param: self.firstFrame: the original first frame of the sequence
        param: self.annotationClass: annotationClass object according to the sequence's characteristics
        param: self.predictionClass: predictionClass object according to the sequence's characteristics
        param: self.step: refers to the number of pacthes extracted in one image
        param: self.currentFrame: refers to the processed frame


        ----------------- Methods -----------------

        self.LaunchSegmentation
        self.LaunchSegmentationFW
        self.extractPatch
        self.InitializationStep
        self.extractPatchFW
        self.getFarWall
        self.ComputeIMT

        '''

        self.desiredYSpacing = 5
        self.fullSq = fullSq

        self.patchHeight = 512
        self.pathWidth = 128

        self.overlay = 8

        self.sequence, self.scale, self.spacing, self.firstFrame = LoadData(sequence=sequencePath,
                                                                            spatialResolution=self.desiredYSpacing,
                                                                            fullSequence=self.fullSq)

        self.annotationClass = annotationClass(annotationPath=pathAnnotation,
                                               nameAnnotation=names,
                                               dimension=self.sequence.shape,
                                               borderPath=pathBorders,
                                               firstFrame=self.firstFrame,
                                               scale=self.scale)



        self.predictionClassFW = predictionClassFW(dimensions = self.sequence.shape,
                                                   patchHeight = self.patchHeight,
                                                   patchWidth = self.pathWidth)
        self.LaunchSegmentationFW()
        saveImage(path = sequencePath, image = self.firstFrame, type='imagesFarWallDetection')
        seg = {'seg': self.annotationClass.mapAnnotation[0,:,0]/self.scale}
        saveMat(path = sequencePath, dic = seg, type = 'farWallMat')
        del self.predictionClassFW


        self.predictionClass = predictionClass(dimensions = self.sequence.shape,
                                                 patchHeight = self.patchHeight,
                                                 patchWidth = self.pathWidth)
        self.sequencePath = sequencePath

        self.step = 0
        self.currentFrame = 0

        self.LaunchSegmentation()

    def LaunchSegmentation(self):


        condition = True

        for frameID in range(self.sequence.shape[0]):

            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0

            # condition give the information if the frame is segmented
            while (condition == True):

                if (self.step == 0): # initialization step
                    x = self.InitializationStep()
                    tmpOverlay = self.overlay

                y = self.annotationClass.yPosition(xLeft=x,
                                                   width=self.pathWidth,
                                                   height=self.patchHeight,
                                                   frameID = frameID)

                self.predictionClass.patches.append({"patch": self.extractPatch(x, y),
                                                     "frameID": frameID,
                                                      "Step": self.step,
                                                      "Overlay": tmpOverlay,
                                                      "(x, y)": (x, y)})
                self.step += 1


                if ((x + self.pathWidth) == self.annotationClass.borders['rightBorder']):                   # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.pathWidth) < (self.annotationClass.borders['rightBorder']):    # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    tmpOverlay = self.overlay

                else:
                    tmp = x + self.pathWidth - self.annotationClass.borders['rightBorder']            # we adjust to reach the right border
                    x -= tmp
                    tmpOverlay = tmp

            condition = True

            self.predictionClass.PredictionMasks()

            tmpMask = self.predictionClass.mapPrediction[frameID, ]

            # if frameID < self.sequence.shape[0]-1 :
            # previousMask[previousMask > 0.5] = 1
            # previousMask[previousMask < 1] = 0
            # self.annotationClass.updateAnnotation(previousMask = tmpMask,
            #                                       frameID = frameID + 1)
            self.annotationClass.updateDynamicProg(previousMask=tmpMask,
                                                   frameID=frameID+1,
                                                   scale=self.scale)

        # saveImage(path = self.sequencePath, image = self.firstFrame/self.scale)
        segLI = {'seg': self.annotationClass.mapAnnotation[1,:,0]/self.scale}
        segMA = {'seg': self.annotationClass.mapAnnotation[1, :, 1]/self.scale}

        # print(2)
        borders = self.annotationClass.borders
        tmp = np.zeros(self.firstFrame.shape + (3,))
        tmp[:, :, 0] = self.firstFrame.copy()
        tmp[:, :, 1] = self.firstFrame.copy()
        tmp[:, :, 2] = self.firstFrame.copy()

        for k in range(borders['leftBorder'], borders['rightBorder']):
            tmp[round(self.annotationClass.mapAnnotation[1,k, 0]/self.scale), k, 0] = 255
            tmp[round(self.annotationClass.mapAnnotation[1, k, 0] / self.scale), k, 1] = 0
            tmp[round(self.annotationClass.mapAnnotation[1, k, 0]/self.scale), k, 2] = 0

            tmp[round(self.annotationClass.mapAnnotation[1, k, 1] / self.scale), k, 0] = 0
            tmp[round(self.annotationClass.mapAnnotation[1, k, 1] / self.scale), k, 1] = 255
            tmp[round(self.annotationClass.mapAnnotation[1, k, 1] / self.scale), k, 2] = 0


        saveImage(path=self.sequencePath, image=tmp, type='segmentedImages')
        saveMat(path = self.sequencePath, dic = segLI, type = 'predictionBoundaries', interface='_IFC3_pred')
        saveMat(path=self.sequencePath, dic=segMA, type='predictionBoundaries', interface='_IFC4_pred')
        scaleDic={'spacing': self.spacing, 'factor': self.scale, 'conversionInMicrometer': self.spacing /self.scale * 10000}
        saveMat(path=self.sequencePath, dic=scaleDic, type='conversion')

    def LaunchSegmentationFW(self):

        img_tmp = self.firstFrame.copy()
        dim = img_tmp.shape
        # --- we first reshape the image to match with the network
        img_tmp = cv2.resize(img_tmp.astype(np.float32), (dim[1], 512), interpolation=cv2.INTER_LINEAR)

        # ---  condition give the information if the segmentation of a frame is over
        condition = True

        # --- we initialize these variables at the beginning of the image
        self.currentFrame = 0
        self.predictionClassFW.patches = []
        self.step = 0

        while (condition == True):

            # --- initialization step
            if (self.step == 0):
                x = self.annotationClass.borders['leftBorder']
                tmpOverlay = self.overlay

            # --- in self.predictionClass.patches are stored the patches at different (x,y) coordinates
            self.predictionClassFW.patches.append({"patch": self.extractPatchFW(x, img_tmp),
                                                 "frameID": self.currentFrame,
                                                  "Step": self.step,
                                                  "Overlay": tmpOverlay,
                                                  "(x)": (x)})
            self.step += 1

            # --- if we reach exactly the last position (on the right)
            if ((x + self.pathWidth) == self.annotationClass.borders['rightBorder']):
                condition = False
            # --- we move the patch from the left to the right with the defined overlay
            elif (x + self.overlay + self.pathWidth) < (self.annotationClass.borders['rightBorder']):
                x += self.overlay
                tmpOverlay = self.overlay

            # --- we adjust the last patch to reach the right border
            else:
                tmp = x + self.pathWidth - self.annotationClass.borders['rightBorder']
                x -= tmp
                tmpOverlay = tmp

        # --- we segment the region under the far wall
        self.predictionClassFW.PredictionMasks()

        self.getFarWall(self.predictionClassFW.mapPrediction)

    def extractPatch(self, x, y):

        """
        We extract the patch from the sequence.

        ---
        return: the desired patch
        """
        return self.sequence[self.currentFrame, y:(y + self.patchHeight), x:(x + self.pathWidth)]

    def InitializationStep(self):

        return self.annotationClass.borders['leftBorder']

    def extractPatchFW(self, x, img):

        """
        We extract the patch from the first frame of the sequence.
        ---
        return: the desired patch
        """
        return img[:, x:(x + self.pathWidth)]

    def getFarWall(self, img):

        # --- we get the bigest connexe region
        img = getBiggestConnexeRegion(img)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(img == 1))

        seed=(round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))

        self.annotationClass.FWAutoInitialization(img = img,
                                                  seed = seed)

        coef = self.firstFrame.shape[0]/512

        # --- DEBUG
        seg = self.annotationClass.mapAnnotation
        seg = seg[0, :, :]*coef
        borders = self.annotationClass.borders

        x = np.linspace(borders['leftBorder'], borders['rightBorder'], borders['rightBorder']-borders['leftBorder']+1)
        regr = np.poly1d(np.polyfit(x, seg[borders['leftBorder']:borders['rightBorder']+1, 0], 3))
        tmp = regr(x)
        tmp[tmp<0]=0
        tmp[tmp>=self.firstFrame.shape[0]] = self.firstFrame.shape[0]-1
        seg[borders['leftBorder']:borders['rightBorder'] + 1, 0], seg[borders['leftBorder']:borders['rightBorder'] + 1, 1] = tmp, tmp




        for k in range(borders['leftBorder'], borders['rightBorder']):
            self.firstFrame[round(seg[k, 0]), k] = 255

        self.annotationClass.mapAnnotation[0,:,] = seg*self.scale
        # print(2)

    def ComputeIMT(self):
        """
        In this function we compute to average value of the intima-media thickness in the region self.annotationClass.bordersROI['leftBorder'] and self.annotationClass.bordersROI['rightBorder']
        ---
        return:  IMTMeanValue, IMTMedianValue
        """
        IMTMeanValue, IMTMedianValue = self.annotationClass.IMT()
        # plotting the points
        # convert in micro meter
        newSpatialResolution = self.spacing /self.scale * 10000 #in micro meter

        plt.subplot(211)
        plt.plot(IMTMeanValue[1:]*newSpatialResolution, "b")
        plt.ylabel('Thickness in $\mu$m')
        plt.legend(['Mean'], loc='lower right')
        plt.subplot(212)
        plt.plot(IMTMedianValue[1:]*newSpatialResolution, "r")
        plt.ylabel('Thickness in $\mu$m')
        plt.xlabel('Frame ID')
        plt.legend(['Median'], loc='lower right')
        plt.savefig("logs/IMTThroughtCardiacCycle")

        print("Width (cm): ", self.spacing*(self.annotationClass.bordersROI['rightBorder']-self.annotationClass.bordersROI['leftBorder']))

        return IMTMeanValue, IMTMedianValue