"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import numpy as np
import os

from package_network.custom_dilated_unet        import custom_dilated_unet
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class predictionClassIMC():

    """ The prediction class contains the trained architecture and performs the following calculations:
    - prediction of masks
    - compute overlay and prediction maps """
    
    def __init__(self, dimensions: tuple, patch_height: int, patch_width: int, borders: dict, p, img=None):

        self.patch_height = patch_height                    # height of the patch
        self.dim = dimensions                               # dimension of the interpolated image
        self.patch_width = patch_width                      # width of the patch
        self.patches = []                                   # list in which extracted patches are stored
        self.predicted_masks = []                           # array in which we store the prediction
        self.final_mask_org = []                            # array in which we store the final combinaison of all prediction
        self.borders = borders                              # left and right borders
        self.map_overlay, self.map_prediction = {}, {}      # dictionaries evolve during the inference phase
        self.img = img

        self.model = self.load_model(p.PATH_MODEL_WALL)

    # ------------------------------------------------------------------------------------------------------------------
    def prediction_masks(self, id: int, pos: dict):
        """ Retrieves patches, then preprocessing is applied and the self.build_maps method reassembles them. """

        patchImg = []
        for i in range(len(self.patches)):
            patchImg.append(self.patches[i]["patch"])
        patches = np.asarray(patchImg)
        patches = self.patch_preprocessing(patches=patches)
        patches = patches[:,:][...,None]
        # --- prediction
        masks = self.model.predict(patches, batch_size=1, verbose=1)
        self.predicted_masks = masks.copy()
        # --- reassemble patches
        self.build_maps(prediction=masks, id=id, pos=pos)

    # ------------------------------------------------------------------------------------------------------------------
    def build_maps(self, prediction: np.ndarray, id: int, pos: dict):
        """ Assembles the patches and predictions to create the overlay map and the prediction map. """

        pred_, overlay_ = np.zeros((pos['max'] - pos['min'], self.dim[2])), np.zeros((pos['max'] - pos['min'], self.dim[2]))
        for i in range(len(self.patches)):
            patch_ = self.patches[i]
            pos_ = patch_["(x, y)"]
            pred_[pos_[1]-pos['min']:(pos_[1]-pos['min']+self.patch_height), pos_[0]:pos_[0] + self.patch_width] = pred_[pos_[1]-pos['min']:pos_[1]-pos['min'] + self.patch_height, pos_[0]:pos_[0] + self.patch_width] + prediction[i,:,:,0]
            overlay_[pos_[1]-pos['min']:pos_[1]-pos['min'] + self.patch_height, pos_[0]:pos_[0] + self.patch_width] = overlay_[pos_[1]-pos['min']:pos_[1]-pos['min'] + self.patch_height, pos_[0]: pos_[0] + self.patch_width] + np.ones((self.patch_height, self.patch_width))

        overlay_[overlay_ == 0] = 1
        pred_ = pred_/overlay_

        self.map_overlay[str(id)] = {"prediction": overlay_.copy(), "offset": pos['min']}
        self.map_prediction[str(id)] = {"prediction": pred_.copy(), "offset": pos['min']}

        # --- for display only
        self.final_mask_org = np.zeros(self.dim[1:])
        mask_tmp_height = pred_.shape[0]
        self.final_mask_org[pos['min']:(pos['min'] + mask_tmp_height),:] = pred_

    # ------------------------------------------------------------------------------------------------------------------
    def load_model(self, model_name: str):
        """ Loads the trained architecture. """

        model = custom_dilated_unet(
            input_shape=(512, 128, 1),
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

        model.load_weights(model_name)
        return model
    
    # ------------------------------------------------------------------------------------------------------------------
    def patch_preprocessing(self, patches: np.ndarray):
        """ Patch preprocessing -> linear histogram between 0 and 255. """
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

# ---------------------------------------------------------------------------------------------------------------------
class predictionClassFW():

    def __init__(self, dimensions, p, img = None):

        self.map_prediction = np.zeros((p.PATCH_HEIGHT, dimensions[2]), dtype=np.float32)
        self.map_overlay = np.zeros((p.PATCH_HEIGHT, dimensions[2]), dtype=np.float32)
        self.patch_height = p.PATCH_HEIGHT
        self.patch_width = p.PATCH_WIDTH
        self.patches = []
        self.img = img
        self.model = self.load_model(os.path.join(p.PATH_TO_LOAD_TRAINED_MODEL_FW, p.MODEL_NAME))

    # ------------------------------------------------------------------------------------------------------------------
    def prediction_masks(self):

        """ In this function we first retreive the pacthes. Then the preprocessed is applied, and the method self.buildMaps is called. """

        patchImg = []

        for i in range(len(self.patches)):
            patchImg.append(self.patches[i]["patch"])

        patches = np.asarray(patchImg)
        patches = self.patch_preprocessing(patches=patches)
        patches = patches[:,:][...,None]

        masks = self.model.predict(patches, batch_size=1, verbose=1)

        self.buildMaps(prediction=masks)

    # ------------------------------------------------------------------------------------------------------------------
    def buildMaps(self,
                  prediction):

        img = np.zeros(self.img.shape + (3,))
        img[..., 0], img[..., 1], img[..., 2] = self.img.copy(), self.img.copy(), self.img.copy()

        for i in range(len(self.patches)):
            data_ = self.patches[i]
            pos_ = data_["(x)"]
            self.map_prediction[:, pos_:pos_+self.patch_width] = self.map_prediction[:, pos_:pos_+self.patch_width] + prediction[i, :, :, 0]
            self.map_overlay[:, pos_:pos_+self.patch_width] = self.map_overlay[:, pos_:pos_+self.patch_width] + np.ones((self.patch_height, self.patch_width))


        # --- Apply this step at the end
        overlay_ = self.map_overlay.copy()
        prediction_ = self.map_prediction.copy()

        overlay_[overlay_ == 0] = 1
        prediction_ = prediction_/overlay_

        self.map_overlay = overlay_
        self.map_prediction = prediction_

    # ------------------------------------------------------------------------------------------------------------------
    def load_model(self, modelName):
        """ Load the model. """
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

    # ------------------------------------------------------------------------------------------------------------------
    def patch_preprocessing(self, patches):
        """ Patch preprocessing, spread grayscale value between 0 and 255. """
        for k in range(patches.shape[0]):
            tmp = patches[k,]
            min = tmp.min()
            tmp = tmp - min
            max = tmp.max()
            tmp = tmp*255/max
            patches[k,] = tmp

        return patches
 # ---------------------------------------------------------------------------------------------------------------------
