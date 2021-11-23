import tensorflow
from tensorflow.keras import regularizers

from caroSegDeepBuildModel.KerasSegmentationFunctions.models.custom_unet import custom_unet
from caroSegDeepBuildModel.KerasSegmentationFunctions.models.satellite_unet import satellite_unet
from caroSegDeepBuildModel.KerasSegmentationFunctions.models.vanilla_unet import vanilla_unet
from caroSegDeepBuildModel.KerasSegmentationFunctions.models.dilated_unet import dilated_unet
from caroSegDeepBuildModel.KerasSegmentationFunctions.models.custom_dilated_unet import custom_dilated_unet
from caroSegDeepBuildModel.KerasSegmentationFunctions.models.custom_dilated_unet_leaky_relu import custom_dilated_unet_leaky_relu
def ModelSelection(ModelName, inputShape, patchWidth=128):

    if ModelName == "custom_unet":
        model = custom_unet(input_shape = inputShape,
                            use_batch_norm = True,
                            num_classes = 1,
                            filters = 32,
                            dropout = 0.2,
                            num_layers = 4,
                            output_activation = 'sigmoid',
                            kernel_regularizer =  None #regularizers.l1(0.001) #regularizers.l1(0.001)
        )

    if ModelName == "satellite_unet":
        model = satellite_unet(input_shape = inputShape,
                               num_classes = 1,
                               output_activation = 'sigmoid',
                               num_layers = 4)

    if ModelName == "vanilla_unet":
        model = vanilla_unet(input_shape=inputShape,
                             num_classes=1,
                             dropout=0.5,
                             filters=64,
                             num_layers=4,
                             output_activation='sigmoid')

    if ModelName == "dilated_unet":
        model = dilated_unet(input_shape=inputShape,
                             mode='cascade',
                             filters=32,
                             n_block=3,
                             n_class=1,
                             output_activation='sigmoid')

    if ModelName == "custom_dilated_unet":

        if patchWidth==128:
            k=2
            b=3
        if patchWidth==64:
            k=3
            b=2

        model = custom_dilated_unet(input_shape=inputShape,
                                    mode='cascade',
                                    # mode = 'parallel',
                                    filters=32,
                                    kernel_size = (3, 3),
                                    n_block=b,
                                    n_pool_col=k,
                                    n_class=1,
                                    output_activation='sigmoid',
                                    SE = None,
                                    # kernel_regularizer=None,
                                    kernel_regularizer = None,
                                    dropout = 0.2)


    if ModelName == "custom_dilated_unet_leaky_relu":
        model = custom_dilated_unet_leaky_relu(input_shape=inputShape,
                                               mode='cascade',
                                               # mode = 'parallel',
                                               filters=32,
                                               kernel_size = (3, 3),
                                               n_block=3,
                                               n_class=1,
                                               output_activation='sigmoid',
                                               SE = None,
                                               kernel_regularizer =  None,
                                               dropout = 0.2) #regularizers.l1(0.001) #regularizers.l1(0.001))

    return model