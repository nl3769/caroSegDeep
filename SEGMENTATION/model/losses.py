import tensorflow
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

import numpy as np


def bce_logdice_loss(y_true, y_pred):
    """
        bce_logdice_loss distance for semantic segmentation.
    """
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))   
 
def bce_dice_loss(y_true, y_pred):
    """
        bce_dice_loss distance for semantic segmentation.
    """
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def DiceBCELoss(targets, inputs, smooth=1e-6):
    """
        DiceBCELoss distance for semantic segmentation.
    """
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

def dice_loss(y_true, y_pred, smooth=1):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])

    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])

    return 1 - K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_bce_loss(y_true, y_pred, smooth=1):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    diceLoss = 1 - dice

    bce = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

    return diceLoss + bce