import tensorflow
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU
import numpy as np

def dice_bce_loss(y_true, y_pred, smooth=1):
    ''' TODO '''
    # --- we compute the Dice
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    dice_loss = 1 - dice
    # --- we compute the BCE
    bce = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

    return dice_loss + bce

def constraint_thickness(y_true, y_pred):
    ''' TODO '''
    y_s = threshold_binarize(y_pred)
    y_s = tensorflow.reduce_sum(y_s, axis=[1])
    y_true = tensorflow.reduce_sum(y_true, axis=[1])

    constraint = tensorflow.reduce_mean(y_s, axis=1) * 5/1000

    return tensorflow.reduce_max(constraint)-0.8

def constraint_MAE(y_true, y_pred):
    ''' TODO '''
    y_s = threshold_binarize(y_pred)
    y_s = tensorflow.reduce_sum(y_s, axis=[1])

    y_true = tensorflow.reduce_sum(y_true, axis=[1])

    res = tensorflow.reduce_mean(tensorflow.abs(y_true - y_s)) * 5/10000

    return res

def threshold_binarize(x, threshold=0.5):
    ''' TODO '''
    ge = tensorflow.greater_equal(x, tensorflow.constant(threshold))
    y = tensorflow.where(ge, x=tensorflow.ones_like(x), y=tensorflow.zeros_like(x))
    return y

def dice_bce_constraint_thickness(y_true, y_pred):
    ''' TODO '''
    dice_bce_loss_ = dice_bce_loss(y_true, y_pred)
    constraint_ = constraint_thickness(y_true, y_pred)

    res = tensorflow.cond(constraint_>0, lambda: tensorflow.add(constraint_, dice_bce_loss_),  lambda: dice_bce_loss_)
    return res

def dice_bce_constraint_MAE(y_true, y_pred):
    ''' TODO '''
    dice_bce_loss_val = dice_bce_loss(y_true, y_pred)
    constraint = constraint_MAE(y_true, y_pred)


    return dice_bce_loss_val + constraint