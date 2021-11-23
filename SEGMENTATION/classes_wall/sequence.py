from classes.prediction import predictionClass
# from classes.annotation import annotationClassTmp
from classes.annotationSemiAuto import annotationClass
from functions.LoadDatas import LoadDICOMSequence

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
        self.extractPatch
        self.ComputeIMT
        '''

        self.desiredYSpacing = 5
        self.fullSq = fullSq

        self.patchHeight = 512
        self.pathWidth = 128

        self.overlay = 15

        self.sequence, self.scale, self.spacing, self.firstFrame = LoadDICOMSequence(sequence=sequencePath,
                                                                                               spatialResolution=self.desiredYSpacing,
                                                                                               fullSequence=self.fullSq)

        self.annotationClass = annotationClass(annotationPath=pathAnnotation,
                                                  nameAnnotation=names,
                                                  dimension=self.sequence.shape,
                                                  borderPath=pathBorders,
                                                  firstFrame=self.firstFrame,
                                                  scale=self.scale)


        self.predictionClass = predictionClass(dimensions=self.sequence.shape,
                                               patchHeight=self.patchHeight,
                                               patchWidth=self.pathWidth)

        self.step = 0
        self.currentFrame = 0



    def LaunchSegmentation(self):

        '''
        We launch the segmentation.
        All the frame of the sequence are segmented through a cine loop. It is possible to segment only the first by setting self.fullSq=False.
        '''

        # ----  condition give the information if the segmentation of a frame is over
        condition = True

        # ---- loop over the sequence
        for frameID in range(self.sequence.shape[0]):

            # ---- we initialize these variables at the beginning of the image
            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0

            while (condition == True):

                # ---- initialization step
                if (self.step == 0):
                    x = self.annotationClass.borders['leftBorder']
                    tmpOverlay = self.overlay

                # ---- y is an integer and the center is centered around the vertical coordinate
                y = self.annotationClass.yPosition(xLeft=x,
                                                   width=self.pathWidth,
                                                   height=self.patchHeight,
                                                   frameID=self.currentFrame)

                # ---- in self.predictionClass.patches are stored the patches at different (x,y) coordinates
                self.predictionClass.patches.append({"patch": self.extractPatch(x, y),
                                                     "frameID": self.currentFrame,
                                                      "Step": self.step,
                                                      "Overlay": tmpOverlay,
                                                      "(x, y)": (x, y)})
                self.step += 1

                # ---- if we reach exactly the last position (on the right)
                if ((x + self.pathWidth) == self.annotationClass.borders['rightBorder']):
                    condition = False
                # ---- we move the patch from the left to the right with the defined overlay
                elif (x + self.overlay + self.pathWidth) < (self.annotationClass.borders['rightBorder']):
                    x += self.overlay
                    tmpOverlay = self.overlay
                # ---- we adjust the last patch to reach the right border
                else:
                    tmp = x + self.pathWidth - self.annotationClass.borders['rightBorder']
                    x -= tmp
                    tmpOverlay = tmp

            condition = True

            # ---- we segment the current frame
            self.predictionClass.PredictionMasks()

            # ---- we extract the borders
            self.annotationClass.updateAnnotation(previousMask = self.predictionClass.mapPrediction[self.currentFrame, ],
                                                  frameID = self.currentFrame + 1)


    def extractPatch(self, x, y):

        """
        We extract the patch from the sequence.

        ----
        return: the desired patch
        """
        return self.sequence[self.currentFrame, y:(y + self.patchHeight), x:(x + self.pathWidth)]

    def ComputeIMT(self):
        """
        In this function we compute to average value of the intima-media thickness in the region self.annotationClass.bordersROI['leftBorder'] and self.annotationClass.bordersROI['rightBorder']

        ----

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