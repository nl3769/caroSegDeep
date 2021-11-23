import os
import cv2

import numpy as np

import h5py

# from functions.SplitData import SplitData

def ImageRead(data_train_m, data_train_i, pathToMasks, pathToImages):
    masks = []
    images = []

    for i in range(len(data_train_m)):
        mask_tmp = cv2.imread(pathToMasks + data_train_m[i], 0)
        img_tmp = cv2.imread(pathToImages + data_train_i[i], 0)

        # mask_tmp = cv2.resize(mask_tmp, (88, 100), interpolation=cv2.INTER_NEAREST)
        # img_tmp = cv2.resize(img_tmp, (88, 100), interpolation=cv2.INTER_NEAREST)

        # mask_tmp = cv2.resize(mask_tmp, (128, 128), interpolation=cv2.INTER_NEAREST)
        # img_tmp = cv2.resize(img_tmp, (128, 128), interpolation=cv2.INTER_NEAREST)

        masks.append(mask_tmp)
        images.append(img_tmp)

    return masks, images

def LoadImages(masks, images, pathToMasks, pathToImages):

    data_train_m = masks["train"]
    data_test_m = masks["test"]
    data_val_m = masks["val"]

    data_train_i = images["train"]
    data_test_i = images["test"]
    data_val_i = images["val"]

    train_m, train_i = ImageRead(data_train_m, data_train_i, pathToMasks, pathToImages)
    test_m, test_i = ImageRead(data_test_m, data_test_i, pathToMasks, pathToImages)
    val_m, val_i = ImageRead(data_val_m, data_val_i, pathToMasks, pathToImages)

    masks = {
        "train": train_m,
        "val": val_m,
        "test": test_m
    }
    images = {
        "train": train_i,
        "val": val_i,
        "test": test_i
    }

    return masks, images

def SortList(images_name, masks_name):
    dim_m = len(masks_name)
    dim_i = len(images_name)
    if(dim_m != dim_i):
        exit("The number of masks is different than the number of images")
    newListImg = []
    for i in range(dim_i):

        tmp_m = masks_name[i]
        tmp_m = tmp_m.replace("patch_mask_", "")
        tmp_i = "patch_image_" + tmp_m
        index = images_name.index(tmp_i)

        newListImg.append(images_name[index])

    return newListImg

def GetFiles(path):
    """
    :param path: path of the analyzed folder
    :return files: a list
    This function allows to get all the files in a folder. It returns a list containing the name of all the files.
    """
    file = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                file.append(entry.name)
    return file

# def LoadData(pathToMasks, pathToImages):
#     images_name = GetFiles(pathToImages)
#     masks_name = GetFiles(pathToMasks)
#
#     images_name = SortList(images_name = images_name,
#                            masks_name = masks_name)
#
#     masks, images = SplitData(masks = masks_name, images = images_name)
#
#     masks, images = LoadImages(masks = masks,
#                                images = images,
#                                pathToMasks = pathToMasks,
#                                pathToImages = pathToImages)
#     return masks, images

def LoadHDF(fileName, keyName, preProcessing):
    hdfFile = h5py.File(fileName, "r")
    datas = hdfFile[keyName]

    mask = datas['masks']
    img = datas['img']
    spatial_resolution = datas['spatial_resolution']

    key_list = list(mask.keys())
    masks_l = []
    img_l = []
    spatialResolution = []

    for i in mask.keys():
        img_l.append(img[i][()])
        masks_l.append(mask[i][()])
        spatialResolution.append(spatial_resolution[i][()])

    masks_l = np.asarray(masks_l)
    img_l = np.asarray(img_l)
    spatialResolution = np.asarray(spatialResolution)

    right_shape_train = masks_l.shape + (1,)

    images_f = np.zeros(right_shape_train)
    masks_f = np.zeros(right_shape_train)

    if preProcessing == True:
        for i in range(len(key_list)):
            tmpMin = np.min(img_l[i,:,:])
            img_l[i, :, :] = (img_l[i, :, :] - tmpMin)
            tmpMax = np.max(img_l[i, :, :])
            coef = 255/tmpMax
            img_l[i, :, :] = img_l[i, :, :]*coef

            images_f[i,:,:,:] = img_l[i,:,:][..., None]
            masks_f[i,:,:,:] = masks_l[i,:,:][..., None]
    else:
        for i in range(len(key_list)):
            images_f[i,:,:,:] = img_l[i,:,:][..., None]
            masks_f[i,:,:,:] = masks_l[i,:,:][..., None]

    return images_f, masks_f, spatialResolution

def LoadHDFMultiInput(fileName, keyName, preProcessing):
    hdfFile = h5py.File(fileName, "r")
    datas = hdfFile[keyName]

    mask = datas['masks']
    imgorg = datas['img']
    imgCLAHE = datas['imgCLAHE']
    spatial_resolution = datas['spatial_resolution']

    key_list = list(mask.keys())
    masks_l = []
    img_l = []
    img_l_CLAHE = []
    spatialResolution = []

    for i in mask.keys():
        img_l.append(imgorg[i][()])
        img_l_CLAHE.append(imgCLAHE[i][()])
        masks_l.append(mask[i][()])
        spatialResolution.append(spatial_resolution[i][()])

    masks_l = np.asarray(masks_l)
    img_l = np.asarray(img_l)
    spatialResolution = np.asarray(spatialResolution)

    right_shape_train = masks_l.shape + (1,)

    images_f = np.zeros(right_shape_train)
    masks_f = np.zeros(right_shape_train)

    if preProcessing == True:
        for i in range(len(key_list)):
            tmpMin = np.min(img_l[i,:,:])
            img_l[i, :, :] = (img_l[i, :, :] - tmpMin)
            tmpMax = np.max(img_l[i, :, :])
            coef = 255/tmpMax
            img_l[i, :, :] = img_l[i, :, :]*coef

            images_f[i,:,:,:] = img_l[i,:,:][..., None]
            masks_f[i,:,:,:] = masks_l[i,:,:][..., None]
    else:
        for i in range(len(key_list)):
            images_f[i,:,:,:] = img_l[i,:,:][..., None]
            masks_f[i,:,:,:] = masks_l[i,:,:][..., None]

    return images_f, masks_f, spatialResolution

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans