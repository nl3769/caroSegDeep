import numpy as np
import imgaug.augmenters as iaa
import random
import matplotlib.pyplot as plt
import tensorflow.keras

class DataGenerator1Channel(tensorflow.keras.utils.Sequence):

    def __init__(self, partitions, labels, dataAugmentation, batch_size=16, dim=(512, 128, 1),  shuffle=True):

        '''
        The DataGenerator1Channel object contains:

        Args:
            partitions (Group): input images
            labels (Group): output images
            dataAugmentation (bool): augment data (:=True) or not (:=False)
            batch_size (int): size of the batch
            dim (tuple): the dimension of the images
            shuffle (bool): to shuffle images at each epoch

        Attributes:
            self.dim(tuple): the dimension of the images
            self.batch_size: batch_size
            self.labels (Group): output images
            self.shuffle (bool): to shuffle images at each epoch
            self.partitions (Group): input images
            self.keys (list): store the patients' name
            self.dataAugmentation (bool): augment data (:=True) or not (:=False)
            self.seq (Sequential): data generator

        Methods:
            augmentation: Add speckle noise to the data
            on_epoch_end: Updates indexes after each epoch
            __data_generation: In this function the data is return by patch (with augmentation or not)
            __getitem__: Denotes the number of batches per epoch
            __len__: Denotes the number of batches per epoch

        '''

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.shuffle = shuffle
        self.partitions = partitions

        self.keys = list(partitions.keys())[0:10]

        self.dataAugmentation = dataAugmentation

        self.seq = iaa.Sequential([iaa.Fliplr(0.5),
                                   iaa.Flipud(0.5),
                                   iaa.Affine(shear=(-2, 2),
                                              rotate=(-5, 5),
                                              translate_px={"y": (-20, 20), "x": (-5, 5)})
                                   ]
                                  )

        self.on_epoch_end()

    def augmentation(self, X, y, speckleNoise = None):
        '''
        Add speckle noise to the data

        Parameters:
            X (np array, float32): input images
            y (np array, float32): associated masks

        Returns:
            x: (np array, float32): input images with added noise
            y (np array, float32): associated masks (unchanged)
        '''

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

    def __len__(self):
        '''
        Denotes the number of batches per epoch

        Returns:
            number of batch per epoch (int)
        '''

        return int(np.ceil(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data with augmentation

        Parameters:
            index (int): allows you to know where you are during the learning phase to ensure that you use all the data during an epoch.

        Returns:
            X (np array, float32): augmented input images
            y (np array, float32): associated masks
        '''

        # --- Generate indexes of the batch
        indexes = self.keys[index * self.batch_size:(index + 1) * self.batch_size]

        # --- Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        '''
        Updates indexes after each epoch

        Parameters:
            None

        Returns:
            None
        '''
        if self.shuffle == True:
            np.random.shuffle(self.keys)

    def __data_generation(self, list_IDs_temp):

        '''
        In this function the data is return by patch (with augmentation or not).

        Parameters:
            list_IDs_temp (list): Contains patients' name to load

        Results:
            X (np array, float32): augmented data
            y (np array, float32): associated masks
        '''

        # --- Initialization
        X = np.empty((len(list_IDs_temp), *self.dim), dtype=np.float32)
        y = np.empty((len(list_IDs_temp), *self.dim), dtype=np.int8)

        # --- Load data
        for i, ID in enumerate(list_IDs_temp):
            # --- Store sample
            X[i,] = self.partitions[ID][()].reshape(self.dim)
            # --- Store class
            y[i,] = self.labels[ID][()].reshape(self.dim)/255

        if self.dataAugmentation:
            # x, y = self.seq(images=X, segmentation_maps=y)
            x, y = self.augmentation(X, y)
            return x, y.astype(np.float32)
        else:
            return X, y.astype(np.float32)

class DataGenerator2Channel(tensorflow.keras.utils.Sequence):
    'Please refer to the document of the class DataGenerator1Channel'
    def __init__(self, partitions, labels, dataAugmentation, batch_size=16, dim=(512, 128),  shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.shuffle = shuffle
        self.partitions = partitions

        # change later
        self.keys = list(labels.keys())
        # self.keys = list(labels.keys())[:16]

        self.dataAugmentation = dataAugmentation

        self.seq = iaa.Sequential([iaa.Fliplr(0.5),
                                   iaa.Flipud(0.5),
                                   iaa.Affine(shear=(-10, 10),
                                              rotate=(-10, 10),
                                              translate_px={"y": (-50, 50)},
                                              scale={"x": (0.9, 1.1), "y": (0.9, 1.1)})
                                   ]
                                  )

        self.on_epoch_end()

    def augmentation(self, X, y, speckleNoise=None):
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.keys[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.keys)

    def __data_generation(self, list_IDs_temp):
        # Initialization

        X = np.empty((len(list_IDs_temp), *self.dim, 2), dtype=np.float32)
        # X_CLAHE = np.empty((len(list_IDs_temp), *self.dim, 2), dtype=np.float32)
        y = np.empty((len(list_IDs_temp), *self.dim, 1), dtype=np.int8)

        # Load data
        for i, ID in enumerate(list_IDs_temp):
        # for i, ID in enumerate(list_IDs_temp[:16]):
            # Store sample
            X[i, :, :, 0] = self.partitions["img"][ID][()].reshape(self.dim)
            X[i, :, :, 1] = self.partitions["CLAHE"][ID][()].reshape(self.dim)

            # Store class
            y[i] = self.labels[ID][()].reshape(self.dim + (1,)) / 255

        if self.dataAugmentation:
            x, y = self.seq(images=X, segmentation_maps=y)
            return x , y.astype(np.float32)
        else:
            return X, y.astype(np.float32)
