'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import numpy as np
from scipy import ndimage
from skimage.measure import label

def get_biggest_connected_region(image):

    img_fill_holes=ndimage.binary_fill_holes(image).astype(int)
    label_image, nbLabels = label(img_fill_holes, return_num=True)

    regionSize=[]

    if nbLabels!=1:
        for k in range(1,nbLabels+1):
            regionSize.append(np.sum(label_image == k))

        regionSize=np.asarray(regionSize)
        idx=np.argmax(regionSize)+1

        label_image[label_image!=idx]=0
        label_image[label_image==idx]=1

        return label_image
    else:
        return img_fill_holes