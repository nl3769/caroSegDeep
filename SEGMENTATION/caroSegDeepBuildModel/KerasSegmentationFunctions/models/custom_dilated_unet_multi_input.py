from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Input, add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def encoder(x, filters=44, n_block=3, kernel_size=(3, 3), activation='relu', n_pool_col = 2):
    skip = []
    skip_pool_col = []

    for i in range(n_pool_col):
        x = Conv2D(filters, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        skip_pool_col.append(x)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)

    for i in range(n_block):
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip, skip_pool_col


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6, kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    dil_f = [1, 2, 3, 4, 5, 6]  # this is for dilated unet

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


def decoder(x, skip1, skip2, skip_pool_col1, skip_pool_col2, filters, n_block=3, kernel_size=(3, 3), activation='relu', n_pool_col = 2):

    for i in reversed(range(n_block)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = concatenate([skip1[i], skip2[i], x])
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters * 2 ** i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)

    for i in reversed(range(n_pool_col)):
        x = UpSampling2D(size=(2, 1))(x)
        x = Conv2D(filters, kernel_size, activation=activation, padding='same')(x)
        x = concatenate([skip_pool_col1[i], skip_pool_col2[i], x])
        x = Conv2D(filters, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)

    return x


def custom_dilated_unet(input_shape,
                        mode,
                        filters,
                        n_block,
                        n_class,
                        output_activation):

    input1 = Input(input_shape)
    input2 = Input(input_shape)

    enc1, skip1, skip_pool_col1 = encoder(input1, filters, n_block)
    enc2, skip2, skip_pool_col2 = encoder(input2, filters, n_block)

    enc = concatenate([enc1, enc2])

    bottle = bottleneck(enc, filters_bottleneck=filters * 2 ** n_block, mode=mode)
    dec = decoder(bottle, skip1, skip2, skip_pool_col1, skip_pool_col2, filters, n_block)
    predictedMap = Conv2D(n_class, (1, 1), activation=output_activation)(dec)

    model = Model(inputs=[input1, input2], outputs=[predictedMap])

    return model
