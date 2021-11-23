import numpy as np
import matplotlib.pyplot as plt
from model.custom_dilated_unet import custom_dilated_unet

from model.Metrics import *
from model.losses import *

from skimage.morphology import closing, opening, square


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class predictionClassFW():

    def __init__(self, dimensions, p, img = None):
        self.mapPrediction = np.zeros((p.PATCH_HEIGHT, dimensions[2]), dtype=np.float32)
        # self.mapPrediction = np.zeros(dimensions, dtype=np.float32)
        self.mapOverlay = np.zeros((p.PATCH_HEIGHT, dimensions[2]), dtype=np.float32)
        self.patchHeight = p.PATCH_HEIGHT
        self.patchWidth = p.PATCH_WIDTH
        self.patches = []
        self.img = img

        self.model = self.LoadModel(os.path.join(p.PATH_TO_LOAD_TRAINED_MODEL_FAR_WALL, 'far_wall.h5'))

    # ------------------------------------------------------------------------------------------------------------------
    def PredictionMasks(self):

        """
        In this function we first retreive the pacthes. Then the preprocessed is applied, and the method self.buildMaps is called
        """

        patchImg = []

        for i in range(len(self.patches)):
            patchImg.append(self.patches[i]["patch"])

        patches = np.asarray(patchImg)
        patches = self.patchPreprocessing(patches=patches)
        patches = patches[:,:][...,None]

        masks = self.model.predict(patches, batch_size=1, verbose=1)

        self.buildMaps(prediction=masks)

    # ------------------------------------------------------------------------------------------------------------------
    def buildMaps(self,
                  prediction):


        # pred_gif = []
        # overlap_gif = []
        # img_gif = []
        # pred_gif.append(self.mapPrediction.copy())
        # overlap_gif.append(self.mapOverlay.copy())
        # img_gif.append(self.img.copy())

        img = np.zeros(self.img.shape + (3,))
        img[..., 0], img[..., 1], img[..., 2] = self.img.copy(), self.img.copy(), self.img.copy()

        for i in range(len(self.patches)):
            tmpData = self.patches[i]
            tmpPos = tmpData["(x)"]
            self.mapPrediction[:, tmpPos:tmpPos+self.patchWidth] = self.mapPrediction[:, tmpPos:tmpPos+self.patchWidth] + prediction[i, :, :, 0]
            self.mapOverlay[:, tmpPos:tmpPos+self.patchWidth] = self.mapOverlay[:, tmpPos:tmpPos+self.patchWidth] + np.ones((self.patchHeight, self.patchWidth))
            # pred_gif.append(self.mapPrediction.copy())
            # overlap_gif.append(self.mapOverlay.copy())
            # img[:, tmpPos:tmpPos + self.patchWidth, 0] = 100
            # img_gif.append(img.copy())

        # import imageio
        # imageio.mimsave('/home/nlaine/Desktop/pred_fw.gif', pred_gif, fps=3)
        # imageio.mimsave('/home/nlaine/Desktop/overlap_fw.gif', overlap_gif, fps=3)
        # imageio.mimsave('/home/nlaine/Desktop/img_fw.gif', img_gif, fps=3)

        # --- Apply this step at the end
        tmpOverlay = self.mapOverlay.copy()
        tmpPrediction = self.mapPrediction.copy()

        tmpOverlay[tmpOverlay == 0] = 1

        tmpPrediction = tmpPrediction/tmpOverlay

        # tmpPrediction[tmpPrediction > 0.5] = 1
        # tmpPrediction[tmpPrediction < 1] = 0

        self.mapOverlay = tmpOverlay
        self.mapPrediction = tmpPrediction

    # ------------------------------------------------------------------------------------------------------------------
    def LoadModel(self, modelName):
        # model = models.load_model(modelName, custom_objects = {'dice_bce_loss': dice_bce_loss ,'iou': iou, 'dice_coef': dice_coef, 'dice_coef_thresholded': dice_coef_thresholded})
        # # model = models.load_model(modelName)
        # # model.summary()
        # return model

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
        return model

    # ------------------------------------------------------------------------------------------------------------------
    def patchPreprocessing(self, patches):

        for k in range(patches.shape[0]):
            tmp = patches[k,]
            min = tmp.min()
            tmp = tmp - min
            max = tmp.max()
            tmp = tmp*255/max
            patches[k,] = tmp

        return patches