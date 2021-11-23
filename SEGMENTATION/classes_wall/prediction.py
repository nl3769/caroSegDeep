import cv2
import numpy as np

from caroSegDeepBuildModel.KerasSegmentationFunctions.losses import *
from caroSegDeepBuildModel.KerasSegmentationFunctions.models import custom_dilated_unet
# from model.custom_dilated_unet import custom_dilated_unet
import matplotlib.pyplot as plt
# from caroSegDeepBuildModel.KerasSegmentationFunctions.losses import dice_bce_loss

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class predictionClass():

    def __init__(self, dimensions, patchHeight, patchWidth, borders, p, flatten=False, img=None):

        self.patchHeight = patchHeight                  # height of the patch
        self.dim = dimensions                           # dimension of the interpolated image
        self.patchWidth = patchWidth                    # width of the patch
        self.patches = []                               # list in which extracted patches are stored
        self.predictedMasks = []                        # array in which we store the prediction
        self.finalMaskOrg = []                             # array in which we store the final combinaison of all prediction
        self.flatten = flatten                          # use if we flatten the ROI
        self.borders = borders                          # left and righ borders
        self.mapOverlay, self.mapPrediction = {}, {}    # dictionaries that evolve during the inference phase
        self.img = img

        self.model = self.load_model(os.path.join(p.PATH_TO_LOAD_TRAINED_MODEL_WALL, 'wall.h5'))

    def prediction_masks(self, id, pos):

        """
        We first retrieve the pacthes. Then the preprocessing is applied and the self.build_maps method reassembles them
        """

        patchImg = []

        for i in range(len(self.patches)):
            patchImg.append(self.patches[i]["patch"])

        patches = np.asarray(patchImg)
        patches = self.patch_preprocessing(patches=patches)
        patches = patches[:,:][...,None]

        masks = self.model.predict(patches, batch_size=1, verbose=1)
        self.predictedMasks = masks.copy()

        if self.flatten == False:
            self.build_maps(prediction=masks, id=id, pos=pos)
        else:
            self.build_mapsFlatten(prediction=masks, id=id, pos=pos)


    def build_maps(self,
                  prediction,
                  id,
                  pos):
        '''
        Assembling the patches and predictions to create the overlay map and the prediction map
        '''
        tmpPred, tmpOverlay = np.zeros((pos['max'] - pos['min'], self.dim[2])), np.zeros((pos['max'] - pos['min'], self.dim[2]))

        for i in range(len(self.patches)):
            tmpData = self.patches[i]
            tmpPos = tmpData["(x, y)"]

            tmpPred[tmpPos[1]-pos['min']:(tmpPos[1]-pos['min']+self.patchHeight), tmpPos[0]:tmpPos[0] + self.patchWidth] = tmpPred[tmpPos[1]-pos['min']:tmpPos[1]-pos['min'] + self.patchHeight, tmpPos[0]:tmpPos[0] + self.patchWidth] + prediction[i,:,:,0]
            tmpOverlay[tmpPos[1]-pos['min']:tmpPos[1]-pos['min'] + self.patchHeight, tmpPos[0]:tmpPos[0] + self.patchWidth] = tmpOverlay[tmpPos[1]-pos['min']:tmpPos[1]-pos['min'] + self.patchHeight, tmpPos[0]: tmpPos[0] + self.patchWidth] + np.ones((self.patchHeight, self.patchWidth))

        tmpOverlay[tmpOverlay == 0] = 1
        tmpPred = tmpPred/tmpOverlay

        self.mapOverlay[str(id)] = {"prediction": tmpOverlay.copy(), "offset": pos['min']}
        self.mapPrediction[str(id)] = {"prediction": tmpPred.copy(), "offset": pos['min']}

        # --- for display only
        self.finalMaskOrg = np.zeros(self.dim[1:])
        mask_tmp_height = tmpPred.shape[0]
        self.finalMaskOrg[pos['min']:(pos['min'] + mask_tmp_height),:] = tmpPred

    def load_model(self, modelName):
        '''
        Load the trained architecture
        '''
        model = custom_dilated_unet(input_shape=(512, 128, 1),
                                    mode='cascade',
                                    filters=32,
                                    kernel_size=(3, 3),
                                    n_block=3,
                                    n_pool_col=2,
                                    n_class=1,
                                    output_activation='sigmoid',
                                    SE=None,
                                    kernel_regularizer=None,
                                    dropout=0.2)

        model.load_weights(modelName)
        return model

    def patch_preprocessing(self, patches):
        '''
        patch preprocessing
        '''

        for k in range(patches.shape[0]):
            tmp = patches[k,]
            min = tmp.min()
            tmp = tmp - min
            max = tmp.max()
            if max == 0:
                max = 0.1
            tmp = tmp*255/max
            patches[k,] = tmp

        return patches
