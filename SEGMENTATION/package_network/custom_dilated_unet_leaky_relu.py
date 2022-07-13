# from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalAveragePooling, Dropout
import tensorflow
from tensorflow.keras.layers import Input, add, concatenate, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import numpy as np

def squeeze_excitation(input, r):
    size_conv = input.shape  # 5 features maps (50*50)

    pooling = GlobalAveragePooling2D()(input)  # (1*1*5)
    tmp = int(size_conv[3] / r)  # tmp = 2
    dens_v1 = Dense(tmp)(pooling)

    relu = Activation('relu')(dens_v1)

    dens_v2 = Dense(size_conv[3])(relu)

    sigmoid = Activation('sigmoid')(dens_v2)

    excitation = tensorflow.reshape(sigmoid, [-1, 1, 1, int(np.shape(input)[-1])])

    return input * excitation


def encoder(x, filters=44, n_block=3, kernel_size=(3, 3), n_pool_col = 2, SE = True, kernel_regularizer=None, dropout=None):
    skip = []
    skip_pool_col = []

    for i in range(n_pool_col):
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        skip_pool_col.append(x)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    for i in range(n_block):
        x = Conv2D(filters * 2 ** i, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        x = Conv2D(filters * 2 ** i, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    return x, skip, skip_pool_col


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6, kernel_size=(3, 3), kernel_regularizer = None, dropout=None):
    dilated_layers = []
    dil_f = [1, 2, 3, 4, 5, 6]  # this is for dilated unet

    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size, padding='same', dilation_rate=dil_f[i], kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = BatchNormalization()(x)
            dilated_layers.append(x)
        return add(dilated_layers)

    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(Conv2D(filters_bottleneck, kernel_size, activation=LeakyReLU(alpha=0.3), padding='same', dilation_rate=2 ** i, kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x))
        return add(dilated_layers)


def decoder(x, skip, skip_pool_col, filters, n_block=3, kernel_size=(3, 3), n_pool_col = 2, SE = True, kernel_regularizer = None, dropout=None):

    for i in reversed(range(n_block)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2 ** i, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = concatenate([skip[i], x])
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        x = Conv2D(filters * 2 ** i, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        x = Conv2D(filters * 2 ** i, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        if dropout > 0:
            x = Dropout(dropout)(x)

    for i in reversed(range(n_pool_col)):
        x = UpSampling2D(size=(2, 1))(x)
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = concatenate([skip_pool_col[i], x])
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        if SE == True:
            x = squeeze_excitation(x, 2)
        if dropout > 0:
            x = Dropout(dropout)(x)
    return x


def custom_dilated_unet_leaky_relu(input_shape,
                                   mode,
                                   filters,
                                   kernel_size,
                                   n_block,
                                   n_class,
                                   output_activation,
                                   SE = False,
                                   kernel_regularizer = None,
                                   dropout = None):


    inputs = Input(input_shape)


    enc, skip, skip_pool_col = encoder(inputs, filters, n_block, kernel_size=kernel_size, SE = SE, kernel_regularizer = kernel_regularizer, dropout = dropout)
    bottle = bottleneck(enc, filters_bottleneck=filters * 2 ** n_block, mode=mode, kernel_regularizer = kernel_regularizer, dropout = dropout)
    dec = decoder(bottle, skip, skip_pool_col, filters, n_block, kernel_size=kernel_size, SE = SE, kernel_regularizer = kernel_regularizer, dropout = dropout)

    classify = Conv2D(n_class, (1, 1), activation=output_activation)(dec)

    model = Model(inputs=[inputs], outputs=[classify])

    return model
