"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import numpy as np
import imgaug.augmenters as iaa
import random
import tensorflow.keras

def get_img_dim(data):
    patient_0 = list(data.keys())[0]
    patch_0 = list(data[patient_0]["img"].keys())[0]

    return data[patient_0]["img"][patch_0][()].shape + (1,)

# ----------------------------------------------------------------------------------------------------------------------
def get_keys(data, keys, pfold):

    id = []

    with open(pfold) as f:
        lines = f.readlines()

    lines = [line.split('.')[0] + '.tiff' for line in lines]

    for patient in lines[:30]:
        if patient in list(data.keys()):
            for id_patch in data[patient][keys[0]].keys():
                img = [[patient], [id_patch]]
                id.append(img)
    return id

# ----------------------------------------------------------------------------------------------------------------------
class dataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, partitions, keys, pfold, data_augmentation, batch_size=16, dim=(512, 128, 1),  shuffle=True):
        """ DataGenerator1Channel apply on the fly data augmentation by using imgaug. """

        self.dim = get_img_dim(data=partitions)
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.data = partitions
        self.keys = get_keys(partitions, keys, pfold)
        self.data_augmentation = data_augmentation
        self.seq = iaa.Sequential([iaa.Fliplr(0.5),
                                   iaa.Flipud(0.5),
                                   iaa.Affine(shear=(-2, 2),
                                              rotate=(-5, 5),
                                              translate_px={"y": (-20, 20), "x": (-5, 5)})])
        self.on_epoch_end()

    # ----------------------------------------------------------------
    def augmentation(self, X, y, speckleNoise = None):
        """ Add speckle noise to the data. """

        x, y = self.seq(images=X, segmentation_maps=y)
        if speckleNoise != None:
            for k in range(X.shape[0]):
                randomVal = random.random()

                if randomVal > 0.5:
                    tmp = x[k, :, :, 0]
                    gauss = np.random.normal(0, 0.1, tmp.size)
                    gauss = gauss.reshape(tmp.shape[0], tmp.shape[1]).astype('float32')
                    noise = tmp + tmp * gauss
                    x[k, :, :, 0] = noise

        return x, y

    # ----------------------------------------------------------------
    def __len__(self):
        """ Denotes the number of batches per epoch. """

        return int(np.ceil(len(self.keys) / self.batch_size))

    # ----------------------------------------------------------------
    def __getitem__(self, index):
        """ Generate one batch of data with augmentation. """

        # --- Generate indexes of the batch
        indexes = self.keys[index * self.batch_size:(index + 1) * self.batch_size]

        # --- Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    # ----------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch. """

        if self.shuffle == True:
            np.random.shuffle(self.keys)

    # ----------------------------------------------------------------
    def __data_generation(self, list_IDs_temp):
        """ Returns data batch (with augmentation or not)."""

        # --- Initialization
        X = np.empty((len(list_IDs_temp), *self.dim), dtype=np.float32)
        y = np.empty((len(list_IDs_temp), *self.dim), dtype=np.int8)

        # --- Load data
        for i, ID in enumerate(list_IDs_temp):
            # --- Store sample
            X[i,] = self.data[ID[0][0]]['img'][ID[1][0]][()].reshape(self.dim)
            # --- Store class
            y[i,] = self.data[ID[0][0]]['masks'][ID[1][0]][()].reshape(self.dim)/255

        if self.data_augmentation:
            x, y = self.augmentation(X, y)
            return x, y.astype(np.float32)
        else:
            return X, y.astype(np.float32)
        
    # ----------------------------------------------------------------
