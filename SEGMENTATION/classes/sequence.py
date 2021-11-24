import os

from classes.prediction import predictionClass
from classes.annotation import annotationClass
from functions.load_datas import load_data

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

        self.sequence, self.scale, self.spacing, self.firstFrame = load_data(sequence = sequencePath, spatialResolution = self.desiredSpatialResolution, fullSequence = self.fullSq, p=p)    # we load the sequence

        self.annotationClass = annotationClass(self.sequence.shape,
                                               pathBorders,
                                               self.firstFrame,
                                               self.scale,
                                               p = p,
                                               patientName=patientName,
                                               overlay=self.overlay)

        self.predictionClass = predictionClass(self.sequence.shape,
                                               self.patchHeight,
                                               self.patchWidth,
                                               self.annotationClass.bordersROI,
                                               p = p,
                                               img=self.sequence[0,])

        self.patch = np.empty((self.patchWidth, self.patchHeight), dtype=np.float32)                                         # the patch
        self.patchPosition = []                                                                                              # patch position (top left)
        self.step = 0                                                                                                        # number of steps
        self.currentFrame = 0
        self.finalMaskAfterPostProcessing = np.zeros(self.sequence.shape[1:])
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
                    x = self.initialization_step()
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

                    self.predictionClass.patches.append({"patch": self.extract_patch(x, y_pos, image = self.sequence[frameID,]),
                                                         "frameID": frameID,
                                                         "Step": self.step,
                                                         "Overlay": tmpOverlay,
                                                         "(x, y)": (x, y_pos)})
                    y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])

                    if y_mean - 128 > 0:
                        y_pos = y_mean + 128
                        self.predictionClass.patches.append({"patch": self.extract_patch(x, y_pos, image = self.sequence[frameID,]),
                                                             "frameID": frameID,
                                                             "Step": self.step,
                                                             "Overlay": tmpOverlay,
                                                             "(x, y)": (x, y_pos)})
                        y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                    if y_mean < self.sequence.shape[1] - 1 :
                        y_pos = y_mean - 128
                        self.predictionClass.patches.append(
                            {"patch": self.extract_patch(x, y_pos, image=self.sequence[frameID,]),
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
                            {"patch": self.extract_patch(x, round(y_inc), image=self.sequence[frameID,]),
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
            self.predictionClass.prediction_masks(id=frameID, pos={"min": min_y, "max": max_y+self.patchHeight})

            tmpMask = self.predictionClass.mapPrediction[str(frameID)]["prediction"]
            mask_tmp = self.annotationClass.update_annotation(previousMask=tmpMask,
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
    def initialization_step(self):
        '''
        return the left border
        '''
        return self.annotationClass.bordersROI['leftBorder']
    # ------------------------------------------------------------------------------------------------------------------------------------
    def extract_patch(self, x, y, image):
        '''
        extracts a patch at a given (x, y) coordinate
        '''
        return image[y:(y + self.patchHeight), x:(x + self.patchWidth)]
    # ------------------------------------------------------------------------------------------------------------------------------------
    def compute_IMT(self, p, patient):

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
