import cv2
import numpy as np

from model.losses import *
from model.custom_dilated_unet import custom_dilated_unet
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
        # if flatten == False:
        #     self.mapPrediction = np.zeros(self.dim, dtype=np.float32)
        #     self.mapOverlay = np.zeros(self.dim, dtype=np.float32)
        # else:
        #     width = self.borders['rightBorder'] - self.borders['leftBorder']
        #     # self.mapPrediction = np.zeros((self.dim[0], self.patchHeight, width), dtype=np.float32)
        #     # self.mapOverlay = np.zeros((self.dim[0], self.patchHeight, width), dtype=np.float32)


        self.model = self.LoadModel(os.path.join(p.PATH_TO_LOAD_TRAINED_MODEL_WALL, 'wall.h5'))

    def PredictionMasks(self, id, pos):

        """
        In this function we first retrieve the pacthes. Then the preprocessing is applied and the self.buildMaps method reassembles them
        """

        patchImg = []

        for i in range(len(self.patches)):
            patchImg.append(self.patches[i]["patch"])

        patches = np.asarray(patchImg)
        patches = self.patchPreprocessing(patches=patches)
        patches = patches[:,:][...,None]

        masks = self.model.predict(patches, batch_size=1, verbose=1)
        self.predictedMasks = masks.copy()

        if self.flatten == False:
            self.buildMaps(prediction=masks, id=id, pos=pos)
        else:
            self.buildMapsFlatten(prediction=masks, id=id, pos=pos)

    def buildMapsFlatten(self,
                         prediction,
                         id,
                         pos):

        tmpFrame = 0

        for i in range(len(self.patches)):

            tmpData = self.patches[i]
            tmpPos = tmpData["(x, y)"]
            tmpFrame = tmpData["frameID"]
            self.mapPrediction[tmpFrame, :, tmpPos[0]:tmpPos[0] + self.patchWidth] +=  prediction[i,:, :, 0]
            self.mapOverlay[tmpFrame, :, tmpPos[0] :tmpPos[0] + self.patchWidth] += np.ones((self.patchHeight, self.patchWidth))

        # --- Apply this step at the end
        tmpOverlay = self.mapOverlay[tmpFrame, :, :]
        tmpPrediction = self.mapPrediction[tmpFrame, :, :]

        tmpOverlay[tmpOverlay == 0] = 1

        tmpPrediction = tmpPrediction / tmpOverlay
        self.finalMaskOrg = tmpPrediction.copy()



        # tmpPrediction[tmpPrediction > 0.5] = 1
        # tmpPrediction[tmpPrediction < 1] = 0

        self.mapOverlay[tmpFrame,] = tmpOverlay
        self.mapPrediction[tmpFrame,] = tmpPrediction


    def buildMaps(self,
                  prediction,
                  id,
                  pos):

        tmpFrame = 0
        # tmpPred, tmpOverlay = np.zeros((self.dim[1], self.dim[2])), np.zeros((self.dim[1], self.dim[2]))

        # tmpPred, tmpOverlay = np.zeros((self.patchHeight+pos['max']-pos['min'], self.dim[2])), np.zeros((self.patchHeight+pos['max']-pos['min'], self.dim[2]))
        tmpPred, tmpOverlay = np.zeros((pos['max'] - pos['min'], self.dim[2])), np.zeros((pos['max'] - pos['min'], self.dim[2]))

        # --- for animation
        # pred_git, overlap_gif, img_tmp_gif = [], [], []
        # pred_git.append(tmpPred.copy())
        # overlap_gif.append(tmpOverlay.copy())
        # img = np.zeros(self.img.shape + (3,))
        # img[...,0], img[...,1], img[...,2] = self.img.copy(), self.img.copy(), self.img.copy()
        # img_tmp_gif.append(img)

        for i in range(len(self.patches)):
            tmpData = self.patches[i]
            tmpPos = tmpData["(x, y)"]

            # tmpPred[tmpPos[1]:tmpPos[1]+self.patchHeight, tmpPos[0]:tmpPos[0]+self.patchWidth] = tmpPred[tmpPos[1]:tmpPos[1]+self.patchHeight, tmpPos[0]:tmpPos[0]+self.patchWidth] + prediction[i, :, :, 0]
            # tmpOverlay[tmpPos[1]:tmpPos[1]+self.patchHeight, tmpPos[0]:tmpPos[0]+self.patchWidth] = tmpOverlay[tmpPos[1]:tmpPos[1]+self.patchHeight, tmpPos[0]:tmpPos[0]+self.patchWidth] + np.ones((self.patchHeight, self.patchWidth))

            tmpPred[tmpPos[1]-pos['min']:(tmpPos[1]-pos['min']+self.patchHeight), tmpPos[0]:tmpPos[0] + self.patchWidth] = tmpPred[tmpPos[1]-pos['min']:tmpPos[1]-pos['min'] + self.patchHeight, tmpPos[0]:tmpPos[0] + self.patchWidth] + prediction[i,:,:,0]
            tmpOverlay[tmpPos[1]-pos['min']:tmpPos[1]-pos['min'] + self.patchHeight, tmpPos[0]:tmpPos[0] + self.patchWidth] = tmpOverlay[tmpPos[1]-pos['min']:tmpPos[1]-pos['min'] + self.patchHeight, tmpPos[0]: tmpPos[0] + self.patchWidth] + np.ones((self.patchHeight, self.patchWidth))


            # --- for animation
            # img[tmpPos[1]:(tmpPos[1] + self.patchHeight), tmpPos[0]:tmpPos[0] + self.patchWidth, 0] = 100
            # img_tmp_gif.append(img.copy())
            # pred_git.append(tmpPred.copy())
            # overlap_gif.append(tmpOverlay.copy())

        # img_gif=[]
        # dim = (400, 300)
        # for n in range(len(img_tmp_gif)):
        #     img_gif.append(cv2.resize(img_tmp_gif[n], dim))

        # import imageio
        # imageio.mimsave('/home/nlaine/Desktop/pred.gif', pred_git, fps=3)
        # imageio.mimsave('/home/nlaine/Desktop/over.gif', overlap_gif, fps=3)
        # imageio.mimsave('/home/nlaine/Desktop/img.gif', img_gif, fps=3)


        tmpOverlay[tmpOverlay == 0] = 1
        tmpPred = tmpPred/tmpOverlay

        # self.finalMaskOrg[str(id)] = {"prediction": tmpPred.copy(), "offset": pos['min']}
        self.mapOverlay[str(id)] = {"prediction": tmpOverlay.copy(), "offset": pos['min']}
        self.mapPrediction[str(id)] = {"prediction": tmpPred.copy(), "offset": pos['min']}

        # --- for display only
        self.finalMaskOrg = np.zeros(self.dim[1:])
        mask_tmp_height = tmpPred.shape[0]
        self.finalMaskOrg[pos['min']:(pos['min'] + mask_tmp_height),:] = tmpPred

    def LoadModel(self, modelName):
        model = custom_dilated_unet(input_shape=(512, 128, 1),
                                    mode='cascade',
                                    # mode = 'parallel',
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
        # model.summary()
        return model

    def patchPreprocessing(self, patches):

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
