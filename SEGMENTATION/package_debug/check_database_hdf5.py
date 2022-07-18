import h5py
import sys
import numpy as np

def main():

    phdf5 = "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/IN_SILICO/DATASET/SILICO_wall.h5"
    dim = (512, 128)
    data = h5py.File(phdf5, 'r+')
    hdf_keys = list(data.keys())

    badimg = []

    for key in hdf_keys:

        img = data[key]['img']
        mask = data[key]['masks']

        img_keys    = list(img.keys())
        # mask_keys   = list(mask.keys())
        for id in img_keys:
            img_ = img[id][()]
            mask_ = mask[id][()]

            dim_img = img_.shape
            dim_mask = mask_.shape

            if dim_img != dim or dim_mask != dim:
                badimg.append(key)

    print(badimg)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()