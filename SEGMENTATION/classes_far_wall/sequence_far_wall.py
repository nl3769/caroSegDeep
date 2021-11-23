import sys

from classes_far_wall.prediction_far_wall import predictionClassFW
from common.annotation import annotationClass
from functions.load_datas import LoadDICOMSequence, LoadData
from functions.processing import getBiggestConnexeRegion
import numpy as np
import cv2
import matplotlib.pyplot as plt

class sequenceClass():

    def __init__(self,
                 sequencePath,
                 pathBorders,
                 patient,
                 p):

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
        param: self.patchWidth: width of the patch
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

        self.desiredYSpacing = p.DESIRED_SPATIAL_RESOLUTION
        self.fullSq = p.PROCESS_FULL_SEQUENCE

        self.patchHeight = p.PATCH_HEIGHT
        self.patchWidth = p.PATCH_WIDTH

        self.overlay = p.OVERLAPPING

        self.sequence, self.scale, self.spacing, self.firstFrame = LoadData(sequence=sequencePath,
                                                                            spatialResolution=self.desiredYSpacing,
                                                                            fullSequence=self.fullSq,
                                                                            p=p)

        self.annotationClass = annotationClass(dimension=self.sequence.shape,
                                               borderPath=pathBorders,
                                               firstFrame=self.firstFrame,
                                               scale=self.scale,
                                               fold=p.FOLD,
                                               overlay=p.OVERLAPPING,
                                               p=p,
                                               patientName=patient)


        self.predictionClassFW = predictionClassFW(dimensions=self.sequence.shape,
                                                   p=p,
                                                   img=cv2.resize(self.firstFrame.astype(np.float32), (self.firstFrame.shape[1], 512), interpolation=cv2.INTER_LINEAR))

        self.step = 0
        self.currentFrame = 0
        self.patch = np.empty((self.patchWidth, self.patchHeight), dtype=np.float32)
        self.patchPosition = []
        self.finalMaskAfterPostProcessing = 0

    # ------------------------------------------------------------------------------------------------------------------
    def LaunchSegmentationFW(self, p):

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
                x = self.annotationClass.bordersROI['leftBorder']
                tmpOverlay = self.overlay

            # --- in self.predictionClass.patches are stored the patches at different (x,y) coordinates
            self.predictionClassFW.patches.append({"patch": self.extractPatchFW(x, img_tmp),
                                                   "frameID": self.currentFrame,
                                                   "Step": self.step,
                                                   "Overlay": tmpOverlay,
                                                   "(x)": (x)})
            self.step += 1

            # --- if we reach exactly the last position (on the right)
            if ((x + self.patchWidth) == self.annotationClass.bordersROI['rightBorder']):
                condition = False
            # --- we move the patch from the left to the right with the defined overlay
            elif (x + self.overlay + self.patchWidth) < (self.annotationClass.bordersROI['rightBorder']):
                x += self.overlay
                tmpOverlay = self.overlay

            # --- we adjust the last patch to reach the right border
            else:
                tmp = x + self.patchWidth - self.annotationClass.bordersROI['rightBorder']
                x -= tmp
                tmpOverlay = tmp

        # --- we segment the region under the far wall
        self.predictionClassFW.PredictionMasks()

        self.getFarWall(self.predictionClassFW.mapPrediction, p)

    # ------------------------------------------------------------------------------------------------------------------
    def extractPatch(self, x, y):

        """
        We extract the patch from the sequence.

        ---
        return: the desired patch
        """
        return self.sequence[self.currentFrame, y:(y + self.patchHeight), x:(x + self.patchWidth)]

    # ------------------------------------------------------------------------------------------------------------------
    def extractPatchFW(self, x, img):

        """
        We extract the patch from the first frame of the sequence.
        ---
        return: the desired patch
        """
        return img[:, x:(x + self.patchWidth)]

    # ------------------------------------------------------------------------------------------------------------------
    def getFarWall(self, img, p):

        # --- we get the bigest connexe region
        img[img>0.5]=1
        img[img<1]=0
        img = getBiggestConnexeRegion(img)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(img == 1))

        seed = (round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))

        self.annotationClass.FWAutoInitialization(img=img,
                                                  seed=seed)

        coef = self.firstFrame.shape[0] / p.PATCH_HEIGHT

        # --- DEBUG
        seg = self.annotationClass.mapAnnotation
        seg = seg[0, :, :] * coef
        borders = self.annotationClass.bordersROI

        tmp_report = seg.copy()

        x = np.linspace(borders['leftBorder'], borders['rightBorder'],
                        borders['rightBorder'] - borders['leftBorder'] + 1)
        regr = np.poly1d(np.polyfit(x, seg[borders['leftBorder']:borders['rightBorder'] + 1, 0], 3))
        tmp = regr(x)
        tmp[tmp < 0] = 0
        tmp[tmp >= self.firstFrame.shape[0]] = self.firstFrame.shape[0] - 1
        seg[borders['leftBorder']:borders['rightBorder'] + 1, 0], seg[borders['leftBorder']:borders['rightBorder'] + 1, 1] = tmp, tmp


        #######################################################################################################################################
        #######################################################################################################################################
        # For report
        # plt.plot(tmp_report[borders['leftBorder']:borders['rightBorder'], 0], linewidth=3)
        # plt.xlabel("Coordonnée x", fontsize=15)
        # plt.ylabel("Coordonnée y", fontsize=15)
        # plt.plot(tmp, linewidth=3)
        # plt.legend(['Contours avant rég.', 'Contours après rég.'], loc='lower left', fontsize=15)
        # plt.savefig("/home/nlaine/Desktop/tmp/contours.png")

        # tmp = self.firstFrame.copy()
        # im = np.zeros(tmp.shape + (3,))
        # im[..., 0] = tmp
        # im[..., 1] = tmp
        # im[..., 2] = tmp

        # for k in range(borders['leftBorder'], borders['rightBorder']):
        #     im[round(seg[k, 0])-2:round(seg[k, 0])+2, k, 0] = 0
        #     im[round(seg[k, 0]) - 2:round(seg[k, 0]) + 2, k, 1] = 255
        #     im[round(seg[k, 0]) - 2:round(seg[k, 0]) + 2, k, 2] = 0

        #######################################################################################################################################
        #######################################################################################################################################

        for k in range(borders['leftBorder'], borders['rightBorder']):
            self.firstFrame[round(seg[k, 0]), k] = 255

        self.annotationClass.mapAnnotation[0, :, ] = seg * self.scale

    # ------------------------------------------------------------------------------------------------------------------
