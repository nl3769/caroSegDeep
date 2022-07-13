"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import h5py

import os
import random

import numpy                        as np
import imageio                      as iio

from tqdm                               import tqdm
from package_utils.load_datas           import load_borders, load_tiff, load_annotation, get_files
from package_utils.patch_extraction     import patch_extraction_wall, patch_extraction_far_wall
from package_utils.folder_handler       import make_dir
from package_utils.check_dir            import chek_dir
from glob                               import glob
from mat4py                             import loadmat
from scipy                              import interpolate

# ----------------------------------------------------------------------------------------------------------------------
def split_patient(data, path, training = 0.7, validation = 0.2):

    keys = list(data.keys())
    random.shuffle(keys)
    dim = len(keys)
    id_training = int(dim * training)
    id_validation = int(dim * training) + int(dim * validation)
    training_patients = keys[0:id_training]
    validation_patients = keys[id_training:id_validation]
    testing_patients = keys[id_validation:]

    pres = os.path.join(path, 'set')
    chek_dir(pres)

    with open(os.path.join(pres, 'training_patients.txt'), 'w') as f:
        for patient in training_patients:
            f.write(patient + '\n')

    with open(os.path.join(pres, 'validation_patients.txt'), 'w') as f:
        for patient in validation_patients:
            f.write(patient + '\n')

    with open(os.path.join(pres, 'testing_patients.txt'), 'w') as f:
        for patient in testing_patients:
            f.write(patient + '\n')

# ----------------------------------------------------------------------------------------------------------------------
def load_seg(id, fname, width):

    LI = loadmat(fname['LI'][id])
    MA = loadmat(fname['MA'][id])

    LI = np.array(LI['LI_val']['seg'])
    MA = np.array(MA['MA_val']['seg'])

    LI[LI < 0] = 0
    MA[MA < 0] = 0

    LI = np.nan_to_num(LI, nan=0)
    MA = np.nan_to_num(MA, nan=0)

    seg_size = LI.shape[0]

    org     = np.linspace(-seg_size/2, seg_size/2, seg_size)
    query   = np.linspace(-width/2, width/2, width)


    F = interpolate.Akima1DInterpolator(org, LI)
    LI = F(query, extrapolate=None)
    F = interpolate.Akima1DInterpolator(org, MA)
    MA = F(query, extrapolate=None)

    return LI, MA

# ----------------------------------------------------------------------------------------------------------------------
def load_data(id, fname):

    I = iio.imread(fname['bmode'][id])

    data = loadmat(fname['I'][id])
    CF = data['image']['CF']

    width = I.shape[1]

    return CF, I, width

# ----------------------------------------------------------------------------------------------------------------------
def get_fname(pdata):

    fname = {'LI'       : [],
             'MA'       : [],
             'I'  : [],
             'bmode'    : []}

    patients = os.listdir(pdata)
    patients.sort()
    substr_bmode = 'bmode_result/RF'
    substr_phantom = 'phantom'

    for patient in patients:
        seq = os.listdir(os.path.join(pdata, patient))
        seq.sort()
        for id_seq in seq:
            pLI = os.path.join(substr_phantom, 'LI.mat')
            pMA = os.path.join(substr_phantom, 'MA.mat')
            pI = sorted(glob(os.path.join(pdata, patient, id_seq, substr_phantom, 'image_information*')))[0]
            pbmode = sorted(glob(os.path.join(pdata, patient, id_seq, substr_bmode, '*_bmode.png')))[0]

            fname['LI'].append(os.path.join(pdata, patient, id_seq, pLI))
            fname['MA'].append(os.path.join(pdata, patient, id_seq, pMA))
            fname['I'].append(pI)
            fname['bmode'].append(pbmode)

    return fname

# ----------------------------------------------------------------------------------------------------------------------
def write_unseen_images(pres: str, substr: str, pimg: str, keys: list):
    """ Write images that cannot be used for training in .txt file. """

    # --- get files
    files = os.listdir(pimg)
    # --- get difference between lists
    diff = list(set(files) - set(keys))
    # --- save images name
    with open(os.path.join(pres, substr + "unseen_images.txt"), "w") as f:
        [f.write(key + '\n') for key in diff]

# ----------------------------------------------------------------------------------------------------------------------
def save_dic_to_HDF5(dic_datas: dict, path: str):
    """ Save patches in .h5 file. """

    f = h5py.File(path, "w")
    for key in dic_datas.keys():
        print(f"The {key} set is being written.")

        tmp_gr = f.create_group(name=key)

        tmp_gr_mask = tmp_gr.create_group(name="masks")                                                             # group for mask
        tmp_gr_img = tmp_gr.create_group(name="img")                                                                # group for patch (img)
        tmp_gr_sr = tmp_gr.create_group(name="spatial_resolution")                                                  # group for spatial resolution (used for hausdorff distance)

        data = dic_datas[key]
        rkey = list(data.keys())[0]

        # --- Writes information
        for patch_id in data[rkey].keys():
            tmp_gr_mask.create_dataset(name=patch_id, data=data["patch_mask"][patch_id], dtype=np.uint8)                     # we convert in uint8 order to gain memory usage
            tmp_gr_img.create_dataset(name=patch_id, data=data["patch_Image_org"][patch_id], dtype=np.float32)
            tmp_gr_sr.create_dataset(name=patch_id, data=data["spatial_resolution"][patch_id], dtype=np.float32)

    f.close()

# ----------------------------------------------------------------------------------------------------------------------
class datasetBuilderIMC():
    def __init__(self, p):
        """ Computes dataset according to CUBS database. """

        self.window = p.PATCH_WIDTH
        self.overlay = p.PATCH_OVERLAY
        self.scale = p.SCALE
        self.dic_datas = {}
        self.im_nb = 0
        self.p = p

    # ------------------------------------------------------------------------------------------------------------------
    def build_data(self):
        """ build_data computes and write the dataset in .h5 file. The h5 contains training, validation and validation set. """

        skipped_sequences=open(os.path.join(self.p.PATH_TO_SKIPPED_SEQUENCES, "skipped_sequences.txt"), "w") # contains images that cannot be incorporated into the data set
        fname = get_fname(self.p.PATH_TO_SEQUENCES)
        # --- loop is required if more than one database is used. It not necessary for CUBS
        nb_I = len(fname['bmode'])
        for img_index in tqdm(range(nb_I), ascii=True, desc='Patch (Mask + image): IN SILICO '):

            # --- load the image and the calibration factor
            spatial_res_y, img_f, width = load_data(img_index, fname)
            spatial_resolution_x = spatial_res_y
            # --- load segmentation
            LI, MA = load_seg(img_index, fname, width)
            nonZerosLI = np.nonzero(LI)
            nonZerosMA = np.nonzero(MA)
            leftB = [np.min(nonZerosLI[0]), np.min(nonZerosMA[0])]
            rightB = [np.max(nonZerosLI[0]), np.max(nonZerosMA[0])]
            borders = [max(leftB) + 10, min(rightB) - 10]
            mat = [LI, MA]
            nameSeq = fname['I'][img_index]
            nameSeq = nameSeq.split('/')[-3]

            # --- extract patch for the current image
            datas_, self.im_nb = patch_extraction_wall(img=img_f,
                                                       manual_del=mat,
                                                       borders=borders,
                                                       width_window=self.window,
                                                       overlay=self.overlay,
                                                       name_seq=nameSeq,
                                                       resize_col=self.scale,
                                                       skipped_sequences=skipped_sequences,
                                                       spatial_res_y=spatial_res_y,
                                                       spatial_res_x=spatial_resolution_x,
                                                       desired_spatial_res=self.p.SPATIAL_RESOLUTION,
                                                       img_nb=self.im_nb)

            # --- add to self.dic_datas if patches were extracted
            if datas_ != "skipped":
                self.dic_datas[nameSeq] = datas_
        skipped_sequences.close()
        print("Total image: ", self.im_nb)
        write_unseen_images(pres=self.p.PATH_TO_SKIPPED_SEQUENCES, substr='wall_', pimg=self.p.PATH_TO_SEQUENCES, keys=list(self.dic_datas.keys()))
        save_dic_to_HDF5(self.dic_datas, os.path.join(self.p.PATH_TO_SAVE_DATASET, "SILICO_wall.h5"))
        split_patient(self.dic_datas, self.p.PATH_TO_SAVE_DATASET)

# ----------------------------------------------------------------------------------------------------------------------
class datasetBuilderFarWall():
    def __init__(self, p):
        """ Computes dataset according to CUBS database. """

        self.window = p.PATCH_WIDTH
        self.overlay = p.PATCH_OVERLAY
        self.scale = p.SCALE
        self.dic_datas = {}
        self.im_nb = 0
        self.p = p

        # --- create directory
        make_dir(p.PATH_TO_SKIPPED_SEQUENCES)
        make_dir(p.PATH_TO_SAVE_DATASET)

    # ------------------------------------------------------------------------------------------------------------------
    def build_data(self):
        """ build_data computes and write the dataset in .h5 file. The h5 contains training, validation and validation set. """

        skipped_sequences = open(os.path.join(self.p.PATH_TO_SKIPPED_SEQUENCES, "skipped_sequences.txt"), "w") # contains images that cannot be incorporated into the data set
        # --- loop is required if more than one database is used. It not necessary for CUBS
        for data_base in self.p.DATABASE_NAME:
            files_cubs = get_files(self.p.PATH_TO_SEQUENCES) # name of all images in self.p.PATH_TO_SEQUENCES
            # --- loop over all the images in the database
            for img_index in tqdm(range(len(files_cubs)), ascii=True, desc='Patch (Mask + image): ' + data_base):
                # --- load the image and the calibration factor
                spatial_res_y, img_f = load_tiff(os.path.join(self.p.PATH_TO_SEQUENCES, files_cubs[img_index]), PATH_TO_CF=self.p.PATH_TO_CF)
                spatial_resolution_x = spatial_res_y
                # --- load left and right borders
                borders = load_borders(os.path.join(self.p.PATH_TO_BORDERS, files_cubs[img_index].split('.')[0] + "_borders.mat"))
                # --- load annotation
                mat = load_annotation(self.p.PATH_TO_CONTOUR, files_cubs[img_index], self.p.EXPERT)
                # --- extract patch for the current image
                datas_, self.im_nb = patch_extraction_far_wall(img=img_f,
                                                               manual_del=mat,
                                                               borders=borders,
                                                               width_window=self.window,
                                                               overlay=self.overlay,
                                                               name_seq=files_cubs[img_index],
                                                               skipped_sequences=skipped_sequences,
                                                               spatial_res_y=spatial_res_y,
                                                               spatial_res_x=spatial_resolution_x,
                                                               img_nb=self.im_nb)
                # --- add to self.dic_datas if patches were extracted
                if datas_ != "skipped":
                    self.dic_datas[files_cubs[img_index]] = datas_
        skipped_sequences.close()
        print("Total image: ", self.im_nb)
        write_unseen_images(pres=self.p.PATH_TO_SAVE_DATASET, substr='far_wall_', pimg = self.p.PATH_TO_SEQUENCES, keys=list(self.dic_datas.keys()))
        save_dic_to_HDF5(self.dic_datas, path=os.path.join(self.p.PATH_TO_SAVE_DATASET, self.p.DATABASE_NAME[0] +  "_far_wall.h5"))

    # ------------------------------------------------------------------------------------------------------------------
    def save_dic_to_HDF5(self, path):
        """ Save patches in .h5 file with train/validation/test sets. """
        f = h5py.File(path, "w")
        for key in self.dic_datas.keys():
            print(f"The {key} set is being written.")

            tmp_gr = f.create_group(name=key)

            tmp_gr_mask = tmp_gr.create_group(name="masks")                                                             # group for mask
            tmp_gr_img = tmp_gr.create_group(name="img")                                                                # group for patch (img)
            tmp_gr_sr = tmp_gr.create_group(name="spatial_resolution")                                                  # group for spatial resolution (used for hausdorff distance)

            data = self.dic_datas[key]
            rkey = list(data.keys())[0]

            # --- Writes information
            for patch_id in data[rkey].keys():
                tmp_gr_mask.create_dataset(name=patch_id, data=data["patch_mask"][patch_id], dtype=np.uint8)            # we convert in uint8 order to gain memory usage
                tmp_gr_img.create_dataset(name=patch_id, data=data["patch_Image_org"][patch_id], dtype=np.float32)
                tmp_gr_sr.create_dataset(name=patch_id, data=data["spatial_resolution"][patch_id], dtype=np.float32)

        f.close()