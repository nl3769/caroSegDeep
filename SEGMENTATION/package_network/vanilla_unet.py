import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Input, concatenate, Cropping2D, MaxPooling2D, SpatialDropout2D, UpSampling2D

def conv2d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer = None):
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(rate=dropout, data_format='channels_last')(c)
    #        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding,
               kernel_regularizer=kernel_regularizer)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def vanilla_unet(
        input_shape,
        num_classes=1,
        dropout=0.5,
        filters=64,
        num_layers=4,
        output_activation='sigmoid'):  # 'sigmoid' or 'softmax'

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2)(x)
        filters = filters * 2  # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)

        ch, cw = get_crop_shape(int_shape(conv), int_shape(x))
        conv = Cropping2D(cropping=(ch, cw))(conv)

        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='same')

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)