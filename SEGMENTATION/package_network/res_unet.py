from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Input, add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from Keras_Segmentation_Functions.losses import dice_loss, bce_dice_loss, weighted_bce_dice_loss
from Keras_Segmentation_Functions.metrics import dice_coef


def encoder_res(x, filters,
                n_block,
                kernel_size,
                activation):
    skip = []
    for i in range(n_block):
        input = x
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters * 2 ** i, kernel_size, activation=None, padding='same')(x)
        x = x+input
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    dil_f = [1, 2, 3, 4, 5, 6]  # this is for dilated unet
    # dil_f =[1,1,1] #this for regular unet remember to change depth=3 and n_blocks=4 when using this
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=dil_f[i])(x)
            x = BatchNormalization()(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2 ** i)(x)
            )
        return add(dilated_layers)


def decoder(x, skip, filters, n_block, kernel_size, activation):
    for i in reversed(range(n_block)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = concatenate([skip[i], x])
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)

    return x


def res_unet(
        input_shape=(1920, 1280, 1),
        filters=32,
        n_block=3,
        n_class=1,
        output_activation='sigmoid'):

    inputs = Input(input_shape)

    enc, skip = encoder_res(inputs = inputs,
                            filters = filters,
                            kernel_size=(3, 3),
                            n_block = n_block,
                            activation='relu')

    dec = decoder(x = enc,
                  skip = skip,
                  filters = filters,
                  n_block = n_block)

    # bottle = bottleneck(enc, filters_bottleneck=filters * 2 ** n_block, mode=mode)
    #
    classify = Conv2D(n_class, (1, 1), activation=output_activation)(dec)

    model = Model(inputs=[inputs], outputs=[classify])

    return model
