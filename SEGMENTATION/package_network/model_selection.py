from package_network.custom_dilated_unet                import custom_dilated_unet

def model_selection(model_name: str, input_shape: tuple, patch_width=128):

    ''' Selection of the desired architecture:
        - custom_unet
        - satellite_unet
        - vanilla_unet
        - dilated_unet
        - custom_dilated_unet
        - custom_dilated_unet_leaky_relu '''


    if model_name == "custom_dilated_unet":

        if patch_width == 128:
            k = 2
            b = 3
        if patch_width == 64:
            k = 3
            b = 2

        model = custom_dilated_unet(
            input_shape = input_shape,
            mode = 'cascade',
            filters = 32,
            kernel_size = (3, 3),
            n_block= b,
            n_pool_col= k,
            n_class = 1,
            output_activation = 'sigmoid',
            SE = None,
            kernel_regularizer = None,
            dropout = 0.2)


    return model