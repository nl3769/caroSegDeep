'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import time

from classes.prediction import predictionClass
from classes.annotation import annotationClass
from functions.load_datas import load_data

class sequenceClass():
    ''' sequenceClass calls all the other classes to perform the calculations. This class contains all the results and runs the sliding window (sliding_window_vertical_scan). '''
    def __init__(self,
                 sequence_path: str,
                 patient_name: str,
                 p):

        self.desired_spatial_res = p.DESIRED_SPATIAL_RESOLUTION
        self.full_seq = p.PROCESS_FULL_SEQUENCE
        self.patch_height = p.PATCH_HEIGHT
        self.patch_width = p.PATCH_WIDTH
        self.overlay = p.OVERLAPPING
        self.sequence, self.scale, self.spacing, self.firstFrame = load_data(sequence=sequence_path, spatial_res=self.desired_spatial_res, full_seq =self.full_seq, p=p)    # we load the sequence
        self.annotationClass = annotationClass(dimension=self.sequence.shape,
                                               first_frame=self.firstFrame,
                                               scale=self.scale,
                                               patient_name=patient_name,
                                               overlay=self.overlay)
        self.predictionClass = predictionClass(self.sequence.shape,
                                               self.patch_height,
                                               self.patch_width,
                                               self.annotationClass.borders_ROI,
                                               p = p,
                                               img=self.sequence[0,])
        self.patch = np.empty((self.patch_width, self.patch_height), dtype=np.float32)
        self.step = 0
        self.current_frame = 0
        self.final_mask_after_post_processing = np.zeros(self.sequence.shape[1:])
    # ------------------------------------------------------------------------------------------------------------------
    def sliding_window_vertical_scan(self):
        ''' At each position, three patches are extracted, and if the difference between the min and the max then the vertical scanning is automacally adjusted. '''
        condition = True
        t = time.time()
        for frame_ID in range(self.sequence.shape[0]):
            self.current_frame = frame_ID
            self.predictionClass.patches = []
            self.step = 0
            y_pos_list = []
            median = (self.annotationClass.map_annotation[frame_ID, :, 0] + self.annotationClass.map_annotation[frame_ID, :, 1]) / 2
            vertical_scanning = True
            # --- condition give the information if the frame is segmented
            while (condition == True):

                if (self.step == 0):  # initialization step
                    x = self.initialization_step()
                    overlay_ = self.overlay

                median_min = np.min(median[x:x+self.patch_width])
                median_max = np.max(median[x:x+self.patch_width])

                y_mean, _, _ = self.annotationClass.yPosition(xLeft=x,
                                                              width=self.patch_width,
                                                              height=self.patch_height,
                                                              map=self.annotationClass.map_annotation[frame_ID,])
                y_pos = y_mean
                # --- by default, we take three patches for a given position x. If this is not enough, the number of patches is adjusted dynamically.
                if 768 > 2*100 + median_max-median_min:
                    self.predictionClass.patches.append({"patch": self.extract_patch(x, y_pos, image = self.sequence[frame_ID,]),
                                                         "frameID": frame_ID,
                                                         "Step": self.step,
                                                         "Overlay": overlay_,
                                                         "(x, y)": (x, y_pos)})
                    y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                    if y_mean - 128 > 0:
                        y_pos = y_mean + 128
                        self.predictionClass.patches.append({"patch": self.extract_patch(x, y_pos, image = self.sequence[frame_ID,]),
                                                             "frameID": frame_ID,
                                                             "Step": self.step,
                                                             "Overlay": overlay_,
                                                             "(x, y)": (x, y_pos)})
                        y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                    if y_mean < self.sequence.shape[1] - 1 :
                        y_pos = y_mean - 128
                        self.predictionClass.patches.append(
                            {"patch": self.extract_patch(x, y_pos, image=self.sequence[frame_ID,]),
                             "frameID": frame_ID,
                             "Step": self.step,
                             "Overlay": overlay_,
                             "(x, y)": (x, y_pos)})
                        y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                # --- if the condition is not met, the artery wall is not fully considered and a vertical scan is applied
                else:
                    y_inc = median_min - 256
                    while(vertical_scanning):

                        if y_inc + 128 > median_max - 128:
                            vertical_scanning=False

                        self.predictionClass.patches.append(
                            {"patch": self.extract_patch(x, round(y_inc), image=self.sequence[frame_ID,]),
                             "frameID": frame_ID,
                             "Step": self.step,
                             "Overlay": overlay_,
                             "(x, y)": (x, round(y_inc))})

                        y_inc+=128
                        y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])

                self.step += 1
                vertical_scanning = True

                if ((x + self.patch_width) == self.annotationClass.borders_ROI['rightBorder']):  # if we reach the last position (on the right)
                    condition = False

                elif (x + self.overlay + self.patch_width) < (self.annotationClass.borders_ROI['rightBorder']):  # we move the patch from the left to the right with an overlay
                    x += self.overlay
                    overlay_ = self.overlay

                else:
                    tmp = x + self.patch_width - self.annotationClass.borders_ROI['rightBorder']  # we adjust to reach the right border
                    x -= tmp
                    overlay_ = tmp

            condition = True

            min_y = min(y_pos_list)
            max_y = max(y_pos_list)
            self.predictionClass.prediction_masks(id=frame_ID, pos={"min": min_y, "max": max_y+self.patch_height})

            mask_ = self.predictionClass.map_prediction[str(frame_ID)]["prediction"]
            mask_tmp = self.annotationClass.update_annotation(previous_mask=mask_,
                                                              frame_ID=frame_ID + 1,
                                                              offset=self.predictionClass.map_prediction[str(frame_ID)]["offset"]).copy()
            mask_tmp_height = mask_tmp.shape[0]
            # --- for display only
            self.final_mask_after_post_processing[self.predictionClass.map_prediction[str(frame_ID)]["offset"]:self.predictionClass.map_prediction[str(frame_ID)]["offset"]+mask_tmp_height,:] = mask_tmp

        exec_t = time.time() - t
        print("execution time: ", exec_t)
    # ------------------------------------------------------------------------------------------------------------------
    def initialization_step(self):
        ''' Returns the left border. '''
        return self.annotationClass.borders_ROI['leftBorder']
    # ------------------------------------------------------------------------------------------------------------------
    def extract_patch(self, x: int, y: int, image: np.ndarray):
        ''' Extracts a patch at a given (x, y) coordinate. '''
        return image[y:(y + self.patch_height), x:(x + self.patch_width)]
    # ------------------------------------------------------------------------------------------------------------------
    def compute_IMT(self, p, patient: str):
        ''' Saves IMT value using annotationClass. '''
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
