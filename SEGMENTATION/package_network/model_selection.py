from package_network.custom_unet                        import custom_unet
from package_network.satellite_unet                     import satellite_unet
from package_network.vanilla_unet                       import vanilla_unet
from package_network.dilated_unet                       import dilated_unet
from package_network.custom_dilated_unet                import custom_dilated_unet
from package_network.custom_dilated_unet_leaky_relu     import custom_dilated_unet_leaky_relu

def model_selection(model_name: str, input_shape: tuple, patch_width=128):

    ''' Selection of the desired architecture:
        - custom_unet
        - satellite_unet
        - vanilla_unet
        - dilated_unet
        - custom_dilated_unet
        - custom_dilated_unet_leaky_relu '''

    if model_name == "custom_unet":
        model = custom_unet(input_shape = input_shape,
                            use_batch_norm = True,
                            num_classes = 1,
                            filters = 32,
                            dropout = 0.2,
                            num_layers = 4,
                            output_activation = 'sigmoid',
                            kernel_regularizer =  None #regularizers.l1(0.001) #regularizers.l1(0.001)
        )

    if model_name == "satellite_unet":
        model = satellite_unet(input_shape = input_shape,
                               num_classes = 1,
                               output_activation = 'sigmoid',
                               num_layers = 4)

    if model_name == "vanilla_unet":
        model = vanilla_unet(input_shape=input_shape,
                             num_classes=1,
                             dropout=0.5,
                             filters=64,
                             num_layers=4,
                             output_activation='sigmoid')

    if model_name == "dilated_unet":
        model = dilated_unet(input_shape=input_shape,
                             mode='cascade',
                             filters=32,
                             n_block=3,
                             n_class=1,
                             output_activation='sigmoid')

    if model_name == "custom_dilated_unet":

        if patch_width==128:
            k=2
            b=3
        if patch_width==64:
            k=3
            b=2

        model = custom_dilated_unet(input_shape=input_shape,
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


    if model_name == "custom_dilated_unet_leaky_relu":
        model = custom_dilated_unet_leaky_relu(input_shape=input_shape,
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