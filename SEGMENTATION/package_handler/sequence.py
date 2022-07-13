"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from package_handler.prediction import predictionClassIMC, predictionClassFW
from package_handler.annotation import annotationClassIMC, annotationClassFW
from package_utils.load_datas import load_data
from package_utils.get_biggest_connected_region import get_biggest_connected_region

class sequenceClassIMC():
    """ sequenceClass calls all the other classes to perform the calculations. This class contains all the results and runs the sliding window (sliding_window_vertical_scan). """
    def __init__(self, sequence_path: str, path_borders: str, patient_name: str, p):

        self.desired_spatial_res = p.DESIRED_SPATIAL_RESOLUTION
        self.full_seq = p.PROCESS_FULL_SEQUENCE
        self.patch_height = p.PATCH_HEIGHT
        self.patch_width = p.PATCH_WIDTH
        self.overlay = p.OVERLAPPING
        self.sequence, self.scale, self.spacing, self.firstFrame = load_data(sequence=sequence_path, spatial_res=self.desired_spatial_res, full_seq =self.full_seq, p=p)    # we load the sequence
        self.annotationClass = annotationClassIMC(dimension=self.sequence.shape,
                                                  first_frame=self.firstFrame,
                                                  scale=self.scale,
                                                  patient_name=patient_name,
                                                  overlay=self.overlay,
                                                  p=p)
        self.predictionClass = predictionClassIMC(self.sequence.shape,
                                                  self.patch_height,
                                                  self.patch_width,
                                                  self.annotationClass.borders_ROI,
                                                  p = p,
                                                  img=self.sequence[0, ])
        self.patch = np.empty((self.patch_width, self.patch_height), dtype=np.float32)
        self.step = 0
        self.current_frame = 0
        self.final_mask_after_post_processing = np.zeros(self.sequence.shape[1:])

    # ------------------------------------------------------------------------------------------------------------------
    def sliding_window_vertical_scan(self):
        """ At each position, three patches are extracted, and if the difference between the min and the max then the vertical scanning is automatically adjusted. """

        condition = True

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

                # --- by default, we take three patches for a given position x. If this is not enough, the number of patches is dynamically adjusted.
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
                    # print(y_pos_list)

                # --- if the condition is not verified, the artery wall is not fully considered and a vertical scan is applied
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
            t = time.time()
            mask_tmp = self.annotationClass.update_annotation(previous_mask=mask_,
                                                              frame_ID=frame_ID + 1,
                                                              offset=self.predictionClass.map_prediction[str(frame_ID)]["offset"]).copy()
            mask_tmp_height = mask_tmp.shape[0]
            # --- for display only
            self.final_mask_after_post_processing[self.predictionClass.map_prediction[str(frame_ID)]["offset"]:self.predictionClass.map_prediction[str(frame_ID)]["offset"]+mask_tmp_height,:] = mask_tmp
            postprocess_time = time.time() - t

        return postprocess_time

    # ------------------------------------------------------------------------------------------------------------------
    def initialization_step(self):
        """ Returns the left border. """

        return self.annotationClass.borders_ROI['leftBorder']

    # ------------------------------------------------------------------------------------------------------------------
    def extract_patch(self, x: int, y: int, image: np.ndarray):
        """ Extracts a patch at a given (x, y) coordinate. """

        return image[y:(y + self.patch_height), x:(x + self.patch_width)]

    # ------------------------------------------------------------------------------------------------------------------
    def compute_IMT(self, p, patient: str):
        """ Saves IMT value using annotationClass. """

        self.annotationClass.mapAnnotation = self.annotationClass.mapAnnotation[1:,]
        IMTMeanValue, IMTMedianValue = self.annotationClass.IMT()
        spatial_res = self.spacing /self.scale * 10000 # in micro meter

        plt.rcParams.update({'font.size': 16})
        plt.subplot(211)
        plt.plot(IMTMeanValue*spatial_res, "b")
        plt.ylabel('Thickness in $\mu$m')
        plt.legend(['Mean'], loc='lower right')
        plt.subplot(212)
        plt.plot(IMTMedianValue*spatial_res, "r")
        plt.ylabel('Thickness in $\mu$m')
        plt.xlabel('Frame ID')
        plt.legend(['Median'], loc='lower right')
        plt.savefig(os.path.join(p.PATH_TO_SAVE_RESULTS_COMPRESSION, patient + '_IMT_compression' + '.png'))
        plt.close()

        return IMTMeanValue, IMTMedianValue

# ----------------------------------------------------------------------------------------------------------------------
class sequenceClassFW():

    def __init__(self,
                 sequence_path: str,
                 path_borders: str,
                 patient_name: str,
                 p):

        self.desired_y_spacing = p.DESIRED_SPATIAL_RESOLUTION
        self.full_seq = p.PROCESS_FULL_SEQUENCE
        self.patch_height = p.PATCH_HEIGHT
        self.patch_width = p.PATCH_WIDTH
        self.overlay = p.OVERLAPPING
        self.sequence, self.scale, self.spacing, self.first_frame = load_data(sequence=sequence_path,
                                                                              spatial_res=self.desired_y_spacing,
                                                                              full_seq=self.full_seq,
                                                                              p=p)
        self.annotationClass = annotationClassFW(dimension=self.sequence.shape,
                                                 borders_path=path_borders,
                                                 first_frame=self.first_frame,
                                                 scale=self.scale,
                                                 overlay=p.OVERLAPPING,
                                                 patient_name=patient_name,
                                                 p=p)

        self.predictionClassFW = predictionClassFW(dimensions=self.sequence.shape,
                                                   p=p,
                                                   img=cv2.resize(self.first_frame.astype(np.float32), (self.first_frame.shape[1], 512), interpolation=cv2.INTER_LINEAR))
        self.step = 0
        self.current_frame = 0
        self.patch = np.empty((self.patch_width, self.patch_height), dtype=np.float32)

    # ------------------------------------------------------------------------------------------------------------------
    def launch_seg_far_wall(self, p):

        img_tmp = self.first_frame.copy()
        dim = img_tmp.shape
        # --- we first reshape the image to match with the network
        img_tmp = cv2.resize(img_tmp.astype(np.float32), (dim[1], 512), interpolation=cv2.INTER_LINEAR)

        # ---  condition give the information if the segmentation of a frame is over
        condition = True

        # --- we initialize these variables at the beginning of the image
        self.current_frame = 0
        self.predictionClassFW.patches = []
        self.step = 0

        while (condition == True):

            # --- initialization step
            if (self.step == 0):
                x = self.annotationClass.borders_ROI['leftBorder']
                overlay_ = self.overlay

            # --- in self.predictionClass.patches are stored the patches at different (x,y) coordinates
            self.predictionClassFW.patches.append({"patch": self.extract_patch_FW(x, img_tmp),
                                                   "frameID": self.current_frame,
                                                   "Step": self.step,
                                                   "Overlay": overlay_,
                                                   "(x)": (x)})
            self.step += 1

            # --- if we reach exactly the last position (on the right)
            if ((x + self.patch_width) == self.annotationClass.borders_ROI['rightBorder']):
                condition = False
            # --- we move the patch from the left to the right with the defined overlay
            elif (x + self.overlay + self.patch_width) < (self.annotationClass.borders_ROI['rightBorder']):
                x += self.overlay
                overlay_ = self.overlay

            # --- we adjust the last patch to reach the right border
            else:
                tmp = x + self.patch_width - self.annotationClass.borders_ROI['rightBorder']
                x -= tmp
                overlay_ = tmp

        # --- we segment the region under the far wall
        self.predictionClassFW.prediction_masks()
        self.get_far_wall(self.predictionClassFW.map_prediction, p)

    # ------------------------------------------------------------------------------------------------------------------
    def extract_patch_FW(self, x, img):
        """ We extract the patch from the first frame of the sequence. """

        return img[:, x:(x + self.patch_width)]

    # ------------------------------------------------------------------------------------------------------------------
    def get_far_wall(self, img, p):
        """ Get far wall. """

        # --- we get the bigest connexe region
        img[img>0.5]=1
        img[img<1]=0
        img = get_biggest_connected_region(img)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(img == 1))
        seed = (round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))
        self.annotationClass.FW_auto_initialization(img=img, seed=seed)
        coef = self.first_frame.shape[0]/p.PATCH_HEIGHT
        seg = self.annotationClass.map_annotation
        seg = seg[0, :, :]*coef
        borders = self.annotationClass.borders_ROI
        x = np.linspace(borders['leftBorder'], borders['rightBorder'],
                        borders['rightBorder'] - borders['leftBorder'] + 1)
        regr = np.poly1d(np.polyfit(x, seg[borders['leftBorder']:borders['rightBorder'] + 1, 0], 3))
        tmp = regr(x)
        tmp[tmp < 0] = 0
        tmp[tmp >= self.first_frame.shape[0]] = self.first_frame.shape[0] - 1
        seg[borders['leftBorder']:borders['rightBorder'] + 1, 0], seg[borders['leftBorder']:borders['rightBorder'] + 1, 1] = tmp, tmp

        self.annotationClass.map_annotation[0, :, ] = seg * self.scale

# ----------------------------------------------------------------------------------------------------------------------