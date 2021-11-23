import tensorflow
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU
import numpy as np

def jaccard_distance(y_true, y_pred, smooth=1):

    """Jaccard distance for semantic segmentation.

    Also known as the intersection-over-union loss.

    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.

    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?

    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.

    # Returns
        The Jaccard distance between the two tensors.

    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)

    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +  K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))   
 
def bce_dice_loss(y_true, y_pred):
    return (binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred))/2

def binary_cross_entropy(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss
    
def weighted_bce_dice_loss(y_true, y_pred):
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss

def DiceBCELoss(targets, inputs, smooth=1e-6):
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
    # --- we compute the Dice
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    diceLoss = 1 - dice
    # --- we compute the BCE
    bce = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

    return diceLoss + bce

def constraint_thickness(y_true, y_pred):
    y_s = threshold_binarize(y_pred)
    y_s = tensorflow.reduce_sum(y_s, axis=[1])
    y_true = tensorflow.reduce_sum(y_true, axis=[1])

    constraint = tensorflow.reduce_mean(y_s, axis=1) * 5/1000

    return tensorflow.reduce_max(constraint)-0.8

def constraint_MAE(y_true, y_pred):
    y_s = threshold_binarize(y_pred)
    y_s = tensorflow.reduce_sum(y_s, axis=[1])

    y_true = tensorflow.reduce_sum(y_true, axis=[1])

    res = tensorflow.reduce_mean(tensorflow.abs(y_true - y_s)) * 5/10000

    return res

def threshold_binarize(x, threshold=0.5):

    ge = tensorflow.greater_equal(x, tensorflow.constant(threshold))
    y = tensorflow.where(ge, x=tensorflow.ones_like(x), y=tensorflow.zeros_like(x))
    return y

def dice_bce_constraint_thickness(y_true, y_pred):

    _dice_bce_loss = dice_bce_loss(y_true, y_pred)
    _constraint = constraint_thickness(y_true, y_pred)

    res = tensorflow.cond(_constraint>0, lambda: tensorflow.add(_constraint, _dice_bce_loss),  lambda: _dice_bce_loss)
    return res

def dice_bce_constraint_MAE(y_true, y_pred):

    _dice_bce_loss = dice_bce_loss(y_true, y_pred)
    _constraint = constraint_MAE(y_true, y_pred)


    return _dice_bce_loss + _constraint