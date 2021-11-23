from classes_far_wall.prediction_far_wall import predictionClassFW
from classes_wall.prediction import predictionClass
# from classes.annotation import annotationClassTmp
from common.annotation import annotationClass
from functions.load_datas import LoadData
from functions.processing import getBiggestConnexeRegion
import numpy as np
import cv2
import matplotlib.pyplot as plt


class sequenceClass():

    def __init__(self,
                 sequencePath,
                 pathBorders,
                 p,
                 patientName):

        '''

        ----------------- Parameters -----------------

        param: sequencePath: path to load the sequence
        param: sequencePath: path to load the annotation.   ##Todo: It will the removed later
        param: names: name of the patient.                  ##Todo: It will the removed later
        param: pathBorders:                                 ##Todo: It will the removed later
        param: fullSq: if the fullSq=True then all the image of the sequence is segmented and only the first frame if fullSq=False.


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
        self.LaunchSegmentationFW
        self.extractPatch
        self.InitializationStep
        self.extractPatchFW
        self.getFarWall
        self.ComputeIMT
        '''

        self.desiredYSpacing = p.DESIRED_SPATIAL_RESOLUTION
        self.fullSq = p.PROCESS_FULL_SEQUENCE

        self.patchHeight = p.PATCH_HEIGHT
        self.patchWidth = p.PATCH_WIDTH
        self.overlay = p.OVERLAPPING

        self.sequencePath = sequencePath

        self.step = 0
        self.currentFrame = 0

        self.sequence, self.scale, self.spacing, self.firstFrame = LoadData(sequence=sequencePath,
                                                                            spatialResolution=self.desiredYSpacing,
                                                                            fullSequence=self.fullSq,
                                                                            p=p)

        self.annotationClass = annotationClass(dimension=self.sequence.shape,
                                               borderPath=pathBorders,
                                               firstFrame=self.firstFrame,
                                               scale=self.scale,
                                               overlay=p.OVERLAPPING,
                                               p=p,
                                               patientName=patientName)

        self.predictionClassFW = predictionClassFW(dimensions = self.sequence.shape,
                                                   p=p)

        self.LaunchSegmentationFW(p)
        # saveImage(path = sequencePath, image = self.firstFrame, type='imagesFarWallDetection')
        # seg = {'seg': self.annotationClass.mapAnnotation[0,:,0]/self.scale}
        # saveMat(path = sequencePath, dic = seg, type = 'farWallMat')
        del self.predictionClassFW


        self.predictionClass = predictionClass(dimensions = self.sequence.shape,
                                               patchHeight=self.patchHeight,
                                               patchWidth=self.patchWidth,
                                               borders=self.annotationClass.borders,
                                               fold=p.FOLD,
                                               p=p)

    # ------------------------------------------------------------------------------------------------------------------
    def launchSegmentation_org(self):

        condition = True

        for frameID in range(self.sequence.shape[0]):

            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0

            # condition give the information if the frame is segmented or not
            while (condition == True):

                if (self.step == 0):  # initialization step
                    x = self.InitializationStep()
                    tmpOverlay = self.overlay

                y, _, _ = self.annotationClass.yPosition(xLeft=x,
                                                         width=self.patchWidth,
                                                         height=self.patchHeight,
                                                         map=self.annotationClass.mapAnnotation[frameID,])

                self.predictionClass.patches.append({"patch": self.extractPatch(x, y, image=self.sequence[frameID,]),
                                                     "frameID": frameID,
                                                     "Step": self.step,
                                                     "Overlay": tmpOverlay,
                                                     "(x, y)": (x, y)})
                self.step += 1

                if ((x + self.patchWidth) == self.annotationClass.borders[
                    'rightBorder']):  # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.patchWidth) < (self.annotationClass.borders[
                    'rightBorder']):  # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    tmpOverlay = self.overlay

                else:
                    tmp = x + self.patchWidth - self.annotationClass.borders[
                        'rightBorder']  # we adjust to reach the right border
                    x -= tmp
                    tmpOverlay = tmp

            condition = True

            self.predictionClass.PredictionMasks()

            tmpMask = self.predictionClass.mapPrediction[frameID,]
            # tmpMask = getBiggestConnexeRegion(tmpMask)
            # self.finalMaskAfterPostProcessing = tmpMask.copy()
            # if frameID < self.sequence.shape[0]-1 :
            self.finalMaskAfterPostProcessing = self.annotationClass.updateAnnotation(previousMask=tmpMask,
                                                                                      frameID=frameID + 1)

        # uncomment in debug mode to vizualize the result

        # segLI = {'seg': self.annotationClass.mapAnnotation[1,:,0]/self.scale}
        # segMA = {'seg': self.annotationClass.mapAnnotation[1, :, 1]/self.scale}
        #
        # borders = self.annotationClass.borders
        # tmp = np.zeros(self.firstFrame.shape + (3,))
        # tmp[:, :, 0] = self.firstFrame.copy()
        # tmp[:, :, 1] = self.firstFrame.copy()
        # tmp[:, :, 2] = self.firstFrame.copy()

        # for k in range(borders['leftBorder'], borders['rightBorder']):
        #     tmp[round(self.annotationClass.mapAnnotation[1,k, 0]/self.scale), k, 0] = 255
        #     tmp[round(self.annotationClass.mapAnnotation[1, k, 0] / self.scale), k, 1] = 0
        #     tmp[round(self.annotationClass.mapAnnotation[1, k, 0]/self.scale), k, 2] = 0
        #
        #     tmp[round(self.annotationClass.mapAnnotation[1, k, 1] / self.scale), k, 0] = 0
        #     tmp[round(self.annotationClass.mapAnnotation[1, k, 1] / self.scale), k, 1] = 255
        #     tmp[round(self.annotationClass.mapAnnotation[1, k, 1] / self.scale), k, 2] = 0

        # saveImage(path=self.sequencePath, image=tmp, type='segmentedImages')
        # saveMat(path = self.sequencePath, dic = segLI, type = 'predictionBoundaries', interface='_IFC3_pred')
        # saveMat(path=self.sequencePath, dic=segMA, type='predictionBoundaries', interface='_IFC4_pred')
        # scaleDic={'spacing': self.spacing, 'factor': self.scale, 'conversionInMicrometer': self.spacing /self.scale * 10000}
        # saveMat(path=self.sequencePath, dic=scaleDic, type='conversion')

    # ------------------------------------------------------------------------------------------------------------------
    def launchSegmentation_vertical_scan(self):

        '''
        At each position, three patches are extracted.
        '''

        condition = True

        for frameID in range(self.sequence.shape[0]):

            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0

            # --- condition give the information if the frame is segmented
            while (condition == True):

                if (self.step == 0):  # initialization step
                    x = self.InitializationStep()
                    tmpOverlay = self.overlay

                y_mean, _, _ = self.annotationClass.yPosition(xLeft=x,
                                                              width=self.patchWidth,
                                                              height=self.patchHeight,
                                                              map=self.annotationClass.mapAnnotation[frameID,])
                if y_mean - 128 > 128:
                    y_pos = y_mean + 128
                else:
                    print("Nope")
                    y_pos=y_mean

                self.predictionClass.patches.append({"patch": self.extractPatch(x, y_pos, image = self.sequence[frameID,]),
                                                     "frameID": frameID,
                                                     "Step": self.step,
                                                     "Overlay": tmpOverlay,
                                                     "(x, y)": (x, y_pos)})

                self.predictionClass.patches.append({"patch": self.extractPatch(x, y_pos, image = self.sequence[frameID,]),
                                                     "frameID": frameID,
                                                     "Step": self.step,
                                                     "Overlay": tmpOverlay,
                                                     "(x, y)": (x, y_pos)})

                if y_mean + 128 < self.sequence.shape[1] - 128:
                    y_pos = y_mean - 128
                else:
                    print("Nope")
                    y_pos = y_mean

                self.predictionClass.patches.append({"patch": self.extractPatch(x, y_pos, image = self.sequence[frameID,]),
                                                     "frameID": frameID,
                                                     "Step": self.step,
                                                     "Overlay": tmpOverlay,
                                                     "(x, y)": (x, y_pos)})

                self.step += 1

                if ((x + self.patchWidth) == self.annotationClass.borders['rightBorder']):  # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.patchWidth) < (self.annotationClass.borders['rightBorder']):  # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    tmpOverlay = self.overlay

                else:
                    tmp = x + self.patchWidth - self.annotationClass.borders['rightBorder']  # we adjust to reach the right border
                    x -= tmp
                    tmpOverlay = tmp

            condition = True

            self.predictionClass.PredictionMasks()

            tmpMask = self.predictionClass.mapPrediction[frameID,]
            # tmpMask = getBiggestConnexeRegion(tmpMask)
            # self.finalMaskAfterPostProcessing = tmpMask.copy()
            # if frameID < self.sequence.shape[0]-1 :
            self.finalMaskAfterPostProcessing = self.annotationClass.updateAnnotation(previousMask=tmpMask,
                                                                                      frameID=frameID + 1).copy()

    # ------------------------------------------------------------------------------------------------------------------
    def launchSegmentation_dynamic_vertical_scan(self):

        '''
        At each position, three patches are extracted, and if the difference between the min and the max then the vertical scanning is automacally adjusted.
        '''

        condition = True

        for frameID in range(self.sequence.shape[0]):

            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0
            y_pos_list = []
            median = (self.annotationClass.mapAnnotation[frameID, :, 0] + self.annotationClass.mapAnnotation[frameID, :, 1]) / 2
            vertical_scanning = True
            # --- condition give the information if the frame is segmented
            while (condition == True):

                if (self.step == 0):  # initialization step
                    x = self.InitializationStep()
                    tmpOverlay = self.overlay

                median_min = np.min(median[x:x+self.patchWidth])
                median_max = np.max(median[x:x+self.patchWidth])

                y_mean, _, _ = self.annotationClass.yPosition(xLeft=x,
                                                              width=self.patchWidth,
                                                              height=self.patchHeight,
                                                              map=self.annotationClass.mapAnnotation[frameID,])

                y_pos = y_mean

                # --- by default, we take three patches for one given x position
                if 768 > 2*100 + median_max-median_min:

                    self.predictionClass.patches.append({"patch": self.extractPatch(x, y_pos, image = self.sequence[frameID,]),
                                                         "frameID": frameID,
                                                         "Step": self.step,
                                                         "Overlay": tmpOverlay,
                                                         "(x, y)": (x, y_pos)})
                    y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])

                    if y_mean - 128 > 0:
                        y_pos = y_mean + 128
                        self.predictionClass.patches.append({"patch": self.extractPatch(x, y_pos, image = self.sequence[frameID,]),
                                                             "frameID": frameID,
                                                             "Step": self.step,
                                                             "Overlay": tmpOverlay,
                                                             "(x, y)": (x, y_pos)})
                        y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                    if y_mean < self.sequence.shape[1] - 1 :
                        y_pos = y_mean - 128
                        self.predictionClass.patches.append(
                            {"patch": self.extractPatch(x, y_pos, image=self.sequence[frameID,]),
                             "frameID": frameID,
                             "Step": self.step,
                             "Overlay": tmpOverlay,
                             "(x, y)": (x, y_pos)})
                        y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                # --- if the condition is not respected while the wall of the artery is not totally considered
                else:
                    y_inc = median_min - 256
                    while(vertical_scanning):

                        if y_inc + 128 > median_max - 128:
                            vertical_scanning=False

                        self.predictionClass.patches.append(
                            {"patch": self.extractPatch(x, round(y_inc), image=self.sequence[frameID,]),
                             "frameID": frameID,
                             "Step": self.step,
                             "Overlay": tmpOverlay,
                             "(x, y)": (x, round(y_inc))})

                        y_inc+=128
                        y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])

                self.step += 1
                vertical_scanning = True

                if ((x + self.patchWidth) == self.annotationClass.bordersROI['rightBorder']):  # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.patchWidth) < (self.annotationClass.bordersROI['rightBorder']):  # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    tmpOverlay = self.overlay

                else:
                    tmp = x + self.patchWidth - self.annotationClass.bordersROI['rightBorder']  # we adjust to reach the right border
                    x -= tmp
                    tmpOverlay = tmp

            condition = True

            min_y = min(y_pos_list)
            max_y = max(y_pos_list)
            self.predictionClass.PredictionMasks(id=frameID, pos={"min": min_y, "max": max_y+self.patchHeight})

            tmpMask = self.predictionClass.mapPrediction[str(frameID)]["prediction"]
            mask_tmp = self.annotationClass.updateAnnotation(previousMask=tmpMask,
                                                             frameID=frameID + 1,
                                                             offset=self.predictionClass.mapPrediction[str(frameID)]["offset"]).copy()
            mask_tmp_height = mask_tmp.shape[0]
            # --- for display only
            self.finalMaskAfterPostProcessing[self.predictionClass.mapPrediction[str(frameID)]["offset"]:self.predictionClass.mapPrediction[str(frameID)]["offset"]+mask_tmp_height,:] = mask_tmp

            # for k in range(self.annotationClass.mapAnnotation[0,:,0].shape[0]):
            #     self.firstFrame[round(self.annotationClass.mapAnnotation[0,k,0]), k] = 255
            # print(2)

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
            if ((x + self.patchWidth) == self.annotationClass.borders['rightBorder']):
                condition = False
            # --- we move the patch from the left to the right with the defined overlay
            elif (x + self.overlay + self.patchWidth) < (self.annotationClass.borders['rightBorder']):
                x += self.overlay
                tmpOverlay = self.overlay

            # --- we adjust the last patch to reach the right border
            else:
                tmp = x + self.patchWidth - self.annotationClass.borders['rightBorder']
                x -= tmp
                tmpOverlay = tmp

        # --- we segment the region under the far wall
        self.predictionClassFW.PredictionMasks()

        self.getFarWall(self.predictionClassFW.mapPrediction, p)

    # ------------------------------------------------------------------------------------------------------------------
    def extractPatch(self, x, y, image):

        return image[y:(y + self.patchHeight), x:(x + self.patchWidth)]

    # ------------------------------------------------------------------------------------------------------------------
    def InitializationStep(self):

        return self.annotationClass.borders['leftBorder']

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
        img = getBiggestConnexeRegion(img)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(img == 1))

        seed=(round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))

        self.annotationClass.FWAutoInitialization(img = img,
                                                  seed = seed)

        coef = self.firstFrame.shape[0]/p.PATCH_HEIGHT

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

    # ------------------------------------------------------------------------------------------------------------------
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