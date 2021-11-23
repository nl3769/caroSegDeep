import os
# from predictionFW import predictionClassFW

from classes_wall.prediction import predictionClass
from common.annotation import annotationClass
from functions.load_datas import LoadData

import numpy as np
import matplotlib.pyplot as plt

class sequenceClass():

    def __init__(self,
                 sequencePath,
                 pathBorders,
                 patientName,
                 p):


        self.desiredSpatialResolution = p.DESIRED_SPATIAL_RESOLUTION
        self.fullSq = p.PROCESS_FULL_SEQUENCE
        self.patchHeight = p.PATCH_HEIGHT
        self.patchWidth = p.PATCH_WIDTH
        self.overlay = p.OVERLAPPING

        self.sequence, self.scale, self.spacing, self.firstFrame = LoadData(sequence = sequencePath, spatialResolution = self.desiredSpatialResolution, fullSequence = self.fullSq, p=p)    # we load the sequence
        self.annotationClass = annotationClass(self.sequence.shape, pathBorders, self.firstFrame, self.scale, p = p, patientName=patientName, overlay=self.overlay)            # self.annotationClass contains the sequence with the annotations
        self.predictionClass = predictionClass(self.sequence.shape, self.patchHeight, self.patchWidth, self.annotationClass.bordersROI, p = p, img=self.sequence[0,])                                 # self.predictionClass contains the sequence with the prediction (overlayMap + preiction Map)

        # for k in range(self.annotationClass.mapAnnotation.shape[2]):
        #     self.sequence[0, round(self.annotationClass.mapAnnotation[0, k, 0]), k] = 255

        self.patch = np.empty((self.patchWidth, self.patchHeight), dtype=np.float32)                                         # the patch
        self.patchPosition = []                                                                                              # patch position (top left)
        self.step = 0                                                                                                        # number of steps
        self.currentFrame = 0

        self.finalMaskAfterPostProcessing = np.zeros(self.sequence.shape[1:])

# ------------------------------------------------------------------------------------------------------------------------------------    

    def launch_segmentation_org(self):
        '''
        Initial implementation of the solution. At each horizontal cordonate i, only one patch is extracted.
        '''
        condition = True

        for frameID in range(self.sequence.shape[0]):

            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0
            y_pos_list = []
            # --- condition give the information if the frame is segmented or not
            while (condition == True):

                # --- initialization step
                if (self.step == 0):
                    x = self.InitializationStep()
                    tmpOverlay = self.overlay

                y = self.annotationClass.yPosition(xLeft=x,
                                                   width=self.patchWidth,
                                                   height=self.patchHeight,
                                                   map=self.annotationClass.mapAnnotation[frameID,])

                self.predictionClass.patches.append({"patch": self.extractPatch(x, y, image=self.sequence[frameID,]),
                                                     "frameID": frameID,
                                                     "Step": self.step,
                                                     "Overlay": tmpOverlay,
                                                     "(x, y)": (x, y)})

                y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][1])

                self.step += 1

                if ((x + self.patchWidth) == self.annotationClass.bordersROI['rightBorder']):                    # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.patchWidth) < (self.annotationClass.bordersROI['rightBorder']):    # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    tmpOverlay = self.overlay

                else:
                    tmp = x + self.patchWidth - self.annotationClass.bordersROI['rightBorder']                   # we adjust to reach the right border
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


# ------------------------------------------------------------------------------------------------------------------------------------    

    def launch_segmentation_sym(self):
        '''
        This implementation is similar to the original one at the difference that e apply a symetry on the borders to avoid edge issu
        '''
        condition = True

        for frameID in range(self.sequence.shape[0]):

            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0
            y_pos_list = []

            workingFrame = self.sequence[frameID,].copy()
            ROI = np.flip(workingFrame[:, self.annotationClass.bordersROI['leftBorder']:self.annotationClass.bordersROI['rightBorder']].copy(), axis=1)
            workingFrame = np.concatenate((ROI, workingFrame[:, self.annotationClass.bordersROI['leftBorder']:self.annotationClass.bordersROI['rightBorder']], ROI), axis=1)
            flipAnn = np.flip(self.annotationClass.mapAnnotation[frameID,self.annotationClass.bordersROI['leftBorder']:self.annotationClass.bordersROI['rightBorder'],], axis=0)
            annotation = np.concatenate((flipAnn, self.annotationClass.mapAnnotation[frameID, self.annotationClass.bordersROI['leftBorder']:self.annotationClass.bordersROI['rightBorder'],], flipAnn))

            for k in range(annotation.shape[0]):
                workingFrame[round(annotation[k,0]), k] = 255

            # condition give the information if the frame is segmented
            while (condition == True):

                if (self.step == 0): # initialization step
                    x = 0
                    tmpOverlay = self.overlay

                y = self.annotationClass.yPosition(xLeft=x,
                                                   width=self.patchWidth,
                                                   height=self.patchHeight,
                                                   map=annotation)

                self.predictionClass.patches.append({"patch": self.extractPatch(x, y, workingFrame),
                                                     "frameID": frameID,
                                                      "Step": self.step,
                                                      "Overlay": tmpOverlay,
                                                      "(x, y)": (x, y)})

                y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][1])

                self.step += 1

                if ((x + self.patchWidth) == workingFrame.shape[1]-1):                    # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.patchWidth) < (workingFrame.shape[1]-1):    # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    tmpOverlay = self.overlay

                else:
                    tmp = x + self.patchWidth - workingFrame.shape[1] + 1                   # we adjust to reach the right border
                    x -= tmp
                    tmpOverlay = tmp

            condition = True

            min_y = min(y_pos_list)
            max_y = max(y_pos_list)
            # max = np.max(median_line[self.annotationClass.bordersROI['leftBorder']:self.annotationClass.bordersROI['rightBorder']])
            self.predictionClass.PredictionMasks(id=frameID, pos={"min": min_y, "max": max_y+self.patchHeight})

            # tmpMask = self.predictionClass.mapPrediction[frameID,]
            tmpMask = self.predictionClass.mapPrediction[str(frameID)]["prediction"]
            mask_tmp = self.annotationClass.updateAnnotation(previousMask=tmpMask,
                                                             frameID=frameID + 1,
                                                             offset=self.predictionClass.mapPrediction[str(frameID)]["offset"]).copy()
            mask_tmp_height = mask_tmp.shape[0]
            # --- for display only
            self.finalMaskAfterPostProcessing[self.predictionClass.mapPrediction[str(frameID)]["offset"]:self.predictionClass.mapPrediction[str(frameID)]["offset"]+mask_tmp_height,:] = mask_tmp


# ------------------------------------------------------------------------------------------------------------------------------------    

    def launch_segmentation_dynamic_vertical_scan(self):

        '''
        At each position, three patches are extracted, and if the difference between the min and the max then the vertical scanning is automacally adjusted.
        '''

        condition = True

        import time
        t = time.time()
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
            self.predictionClass.PredictionMasks(id=frameID, pos={"min": min_y,
                                                                  "max": max_y+self.patchHeight})

            tmpMask = self.predictionClass.mapPrediction[str(frameID)]["prediction"]
            mask_tmp = self.annotationClass.updateAnnotation(previousMask=tmpMask,
                                                             frameID=frameID + 1,
                                                             offset=self.predictionClass.mapPrediction[str(frameID)]["offset"]).copy()
            mask_tmp_height = mask_tmp.shape[0]
            # --- for display only
            self.finalMaskAfterPostProcessing[self.predictionClass.mapPrediction[str(frameID)]["offset"]:self.predictionClass.mapPrediction[str(frameID)]["offset"]+mask_tmp_height,:] = mask_tmp

            # for k in range(self.annotationClass.mapAnnotation[0,:,0].shape[0]):
            #     self.firstFrame[round(self.annotationClass.mapAnnotation[1,k,0]/self.scale), k] = 255
            #     self.firstFrame[round(self.annotationClass.mapAnnotation[1, k, 1]/self.scale), k] = 255
        exec_t = time.time() - t
        print("execution time: ", exec_t)
# ------------------------------------------------------------------------------------------------------------------------------------    

    def launch_segmentation_vertical_scan(self):

        '''
        At each position, three patches are extracted
        '''

        condition = True

        for frameID in range(self.sequence.shape[0]):

            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0
            y_pos_list = []
            median = (self.annotationClass.mapAnnotation[frameID, :, 0] + self.annotationClass.mapAnnotation[
                                                                          frameID, :, 1]) / 2
            vertical_scanning = True
            # --- condition give the information if the frame is segmented
            while (condition == True):

                if (self.step == 0):  # initialization step
                    x = self.InitializationStep()
                    tmpOverlay = self.overlay

                median_min = np.min(median[x:x + self.patchWidth])
                median_max = np.max(median[x:x + self.patchWidth])

                y_mean, _, _ = self.annotationClass.yPosition(xLeft=x,
                                                              width=self.patchWidth,
                                                              height=self.patchHeight,
                                                              map=self.annotationClass.mapAnnotation[frameID,])

                y_pos = y_mean

                # --- by default, we take three patches for one given x position
                self.predictionClass.patches.append(
                    {"patch": self.extractPatch(x, y_pos, image=self.sequence[frameID,]),
                     "frameID": frameID,
                     "Step": self.step,
                     "Overlay": tmpOverlay,
                     "(x, y)": (x, y_pos)})
                y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])

                if y_mean - 128 > 0:
                    y_pos = y_mean + 128
                    self.predictionClass.patches.append(
                        {"patch": self.extractPatch(x, y_pos, image=self.sequence[frameID,]),
                         "frameID": frameID,
                         "Step": self.step,
                         "Overlay": tmpOverlay,
                         "(x, y)": (x, y_pos)})
                    y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                if y_mean < self.sequence.shape[1] - 1:
                    y_pos = y_mean - 128
                    self.predictionClass.patches.append(
                        {"patch": self.extractPatch(x, y_pos, image=self.sequence[frameID,]),
                         "frameID": frameID,
                         "Step": self.step,
                         "Overlay": tmpOverlay,
                         "(x, y)": (x, y_pos)})
                    y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])


                self.step += 1
                vertical_scanning = True

                if ((x + self.patchWidth) == self.annotationClass.bordersROI[
                    'rightBorder']):  # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.patchWidth) < (self.annotationClass.bordersROI[
                    'rightBorder']):  # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    tmpOverlay = self.overlay

                else:
                    tmp = x + self.patchWidth - self.annotationClass.bordersROI[
                        'rightBorder']  # we adjust to reach the right border
                    x -= tmp
                    tmpOverlay = tmp

            condition = True

            min_y = min(y_pos_list)
            max_y = max(y_pos_list)
            self.predictionClass.PredictionMasks(id=frameID, pos={"min": min_y, "max": max_y + self.patchHeight})

            tmpMask = self.predictionClass.mapPrediction[str(frameID)]["prediction"]
            mask_tmp = self.annotationClass.updateAnnotation(previousMask=tmpMask,
                                                             frameID=frameID + 1,
                                                             offset=
                                                             self.predictionClass.mapPrediction[str(frameID)]["offset"]).copy()
            mask_tmp_height = mask_tmp.shape[0]
            # --- for display only
            self.finalMaskAfterPostProcessing[self.predictionClass.mapPrediction[str(frameID)]["offset"]:
                                              self.predictionClass.mapPrediction[str(frameID)]["offset"] + mask_tmp_height, :] = mask_tmp

            # for k in range(self.annotationClass.mapAnnotation[0,:,0].shape[0]):
            #     self.firstFrame[round(self.annotationClass.mapAnnotation[0,k,0]), k] = 255
            # print(2)

    # ------------------------------------------------------------------------------------------------------------------------------------

    # TODO: Review the code above
    def flattenSegmentation(self):

        condition = True

        # --- we first flatten the image
        widthROI = self.annotationClass.borders['rightBorder'] - self.annotationClass.borders['leftBorder']
        flattenImage = np.zeros((512, widthROI))

        ################################################################################################################################################################
        ################################################################################################################################################################
        ################################################################################################################################################################
        # --- debug: I load the expert's annotation and then I flatten the image
        # namePatient = "clin_0201_R.mat"
        # pathToAnnotation = "../../../Data/Contours/MEIBURGER/" + namePatient.split('.')[0]
        #
        # dataPathIFC3 = pathToAnnotation + "_IFC3_A1.mat"
        # dataPathIFC4 = pathToAnnotation + "_IFC4_A1.mat"

        # import scipy.io
        # mat_IFC3 = scipy.io.loadmat(dataPathIFC3)
        # mat_IFC3 = mat_IFC3['seg']
        #
        # mat_IFC4 = scipy.io.loadmat(dataPathIFC4)
        # mat_IFC4 = mat_IFC4['seg']
        #
        # for k in range(self.sequence.shape[2]):
        #     if not np.isnan(mat_IFC4[0, k]):
        #         self.sequence[0, round(mat_IFC4[0, k]*self.scale), k] = 255
        #     # self.sequence[0, round(mat_IFC3[0, k]*self.scale), k] = 255

        ################################################################################################################################################################
        ################################################################################################################################################################
        ################################################################################################################################################################

        # LoadAnnotation

        for k in range(self.annotationClass.borders['leftBorder'], self.annotationClass.borders['rightBorder']):
            flattenImage[255:, self.annotationClass.borders['leftBorder']-k] = self.sequence[0, round(self.annotationClass.mapAnnotation[0, k, 0]):round(self.annotationClass.mapAnnotation[0, k, 0])+257, k]
            flattenImage[0:255, self.annotationClass.borders['leftBorder'] - k] = self.sequence[0, round(self.annotationClass.mapAnnotation[0, k, 0]) - 255:round(self.annotationClass.mapAnnotation[0, k, 0]), k]

        ################################################################################################################################################################
        ################################################################################################################################################################
        ################################################################################################################################################################

        flattenIFC4 = np.where(flattenImage == 255)

        tmpIFC4 = np.zeros(flattenImage.shape[1])
        for k in range(flattenIFC4[1][0], flattenIFC4[1][-1]):
            tmpIFC4[k] = flattenIFC4[0][k]

        self.annotationClass.mapAnnotation[1, self.annotationClass.borders['leftBorder']:self.annotationClass.borders['rightBorder'], 0] = self.annotationClass.mapAnnotation[0, self.annotationClass.borders['leftBorder']:self.annotationClass.borders['rightBorder'], 0] + mat_IFC4[0, self.annotationClass.borders['leftBorder']:self.annotationClass.borders['rightBorder']] - 255

        # plt.imsave("/home/nlaine/Desktop/test_000.png", self.sequence[0, :], cmap='gray')

        for k in range(self.annotationClass.borders['leftBorder'], self.annotationClass.borders['rightBorder']):
            if not np.isnan(self.annotationClass.mapAnnotation[1, k, 0]):
                self.sequence[0, round(self.annotationClass.mapAnnotation[1, k, 0]), k] = 255

        # plt.imsave("/home/nlaine/Desktop/test_001.png", self.sequence[0, :], cmap='gray')

        ################################################################################################################################################################
        ################################################################################################################################################################
        ################################################################################################################################################################

        for frameID in range(self.sequence.shape[0]):

            self.currentFrame = frameID
            self.predictionClass.patches = []
            self.step = 0

            # condition give the information if the frame is segmented
            x = 0
            tmpOverlay = self.overlay
            while (condition == True):

                y=0

                self.predictionClass.patches.append({"patch": flattenImage[:, x:x+self.patchWidth],
                                                     "frameID": frameID,
                                                     "Step": self.step,
                                                     "Overlay": tmpOverlay,
                                                     "(x, y)": (x, y)})
                self.step += 1

                if (x + self.patchWidth) == flattenImage.shape[1]:  # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.patchWidth) < flattenImage.shape[1]:  # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    tmpOverlay = self.overlay

                else:
                    tmp = x + self.patchWidth - flattenImage.shape[1]  # we adjust to reach the right border
                    x -= tmp
                    tmpOverlay = tmp

            condition = True

            self.predictionClass.PredictionMasks(id=frameID)


            tmpMask = self.predictionClass.mapPrediction[frameID,]

            self.annotationClass.updateAnnotationFlatten(prediction=tmpMask,
                                                         frameID=frameID+1,
                                                         scale=self.scale,
                                                         tmp=self.sequence.copy())
            # self.finalMaskAfterPostProcessing = self.annotationClass.updateAnnotation(previousMask=tmpMask,
            #                                                                           frameID=frameID + 1)


# ------------------------------------------------------------------------------------------------------------------------------------    

    def InitializationStep(self):

        return self.annotationClass.bordersROI['leftBorder']


# ------------------------------------------------------------------------------------------------------------------------------------    

    def extractPatch(self, x, y, image):

        return image[y:(y + self.patchHeight), x:(x + self.patchWidth)]


# ------------------------------------------------------------------------------------------------------------------------------------    

    def computeIMT(self, p, patient):

        self.annotationClass.mapAnnotation = self.annotationClass.mapAnnotation[1:,]
        IMTMeanValue, IMTMedianValue = self.annotationClass.IMT()
        # plotting the points
        # convert in micro meter
        newSpatialResolution = self.spacing /self.scale * 10000 # in micro meter

        plt.rcParams.update({'font.size': 16})
        plt.subplot(211)
        plt.plot(IMTMeanValue*newSpatialResolution, "b")
        plt.ylabel('Thickness in $\mu$m')
        plt.legend(['Mean'], loc='lower right')
        plt.subplot(212)
        plt.plot(IMTMedianValue*newSpatialResolution, "r")
        plt.ylabel('Thickness in $\mu$m')
        plt.xlabel('Frame ID')
        plt.legend(['Median'], loc='lower right')
        plt.savefig(os.path.join(p.PATH_TO_SAVE_RESULTS_COMPRESSION, patient + '_IMT_compression' + '.png'))
        plt.close()

        return IMTMeanValue, IMTMedianValue
