import matplotlib.pyplot as plt
import numpy as np

def displayImagesList(seqI, seqM, firstFrameNb, lastFrameNb):
    '''
    This function displays k = lastFrameNb - firstFrameNb images of a np.array which has a shape of (x,y,z,1).
    :param seqI: Images
    :param seqM: Masks
    :param firstFrameNb: first frame we want to display
    :param lastFrameNb: last frame we want to display
    :return: nothing but display k images
    '''
    images = seqI[firstFrameNb:lastFrameNb, :, :, :]
    masks = seqM[firstFrameNb:lastFrameNb, :, :, :]
    dim = images.shape

    for i in range(dim[0]):
        # plt.figure()
        fig, axs = plt.subplots(1,2)
        fig.suptitle('horizontally stacked subplots')
        axs[0].imshow(images[i, :, :, :], cmap='gray')
        axs[1].imshow(masks[i, :, :, :], cmap='gray')

    plt.show()

def displayImagesTwoChannels(seqI, seqM, firstFrameNb, lastFrameNb):
    '''
    This function displays k = lastFrameNb - firstFrameNb images of a np.array which has a shape of (x,y,z,1).
    :param seqI: Images
    :param seqM: Masks
    :param firstFrameNb: first frame we want to display
    :param lastFrameNb: last frame we want to display
    :return: nothing but display k images
    '''
    images = seqI[firstFrameNb:lastFrameNb, :, :, 0]
    masks = seqM[firstFrameNb:lastFrameNb, :, :, :]
    dim = images.shape

    for i in range(dim[0]):
        # plt.figure()
        fig, axs = plt.subplots(1,2)
        fig.suptitle('horizontally stacked subplots')
        axs[0].imshow(images[i, :, :, :], cmap='gray')
        axs[1].imshow(masks[i, :, :, :], cmap='gray')

    plt.show()