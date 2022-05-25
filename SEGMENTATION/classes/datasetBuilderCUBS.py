"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import h5py

from tqdm import tqdm
import os
import numpy as np
from functions.load_datas import load_borders, load_tiff, load_annotation, get_files
from functions.patch_extraction import patch_extraction_wall, patch_extraction_far_wall
from functions.split_data import split_data_fold
from functions.folder_handler import make_dir

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
    """ Save patches in .h5 file with train/validation/test sets. """

    # def dic_datas(self, path):
    """ Save patches in .h5 file with train/validation/test sets. """
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
        # --- loop is required if more than one database is used. It not necessary for CUBS
        for data_base in self.p.DATABASE_NAME:
            files_cubs = get_files(self.p.PATH_TO_SEQUENCES) # name of all images in self.p.PATH_TO_SEQUENCES
            # --- loop over all the images in the database
            # for img_index in tqdm(range(len(files_cubs)), ascii=True, desc='Patch (Mask + image): ' + data_base):
            for img_index in tqdm(range(len(files_cubs)), ascii=True, desc='Patch (Mask + image): ' + data_base):
                # --- load the image and the calibration factor
                spatial_res_y, img_f = load_tiff(os.path.join(self.p.PATH_TO_SEQUENCES, files_cubs[img_index]), PATH_TO_CF=self.p.PATH_TO_CF)
                spatial_resolution_x = spatial_res_y
                # --- load left and right borders
                borders = load_borders(os.path.join(self.p.PATH_TO_BORDERS, files_cubs[img_index].split('.')[0] + "_borders.mat"))
                # --- load annotation
                mat = load_annotation(self.p.PATH_TO_CONTOUR, files_cubs[img_index], self.p.EXPERT)
                # --- extract patch for the current image
                datas_, self.im_nb = patch_extraction_wall(img=img_f,
                                                           manual_del=mat,
                                                           borders=borders,
                                                           width_window=self.window,
                                                           overlay=self.overlay,
                                                           name_seq=files_cubs[img_index],
                                                           resize_col=self.scale,
                                                           skipped_sequences=skipped_sequences,
                                                           spatial_res_y=spatial_res_y,
                                                           spatial_res_x=spatial_resolution_x,
                                                           desired_spatial_res=self.p.SPATIAL_RESOLUTION,
                                                           img_nb=self.im_nb)
                # --- add to self.dic_datas if patches were extracted
                if datas_ != "skipped":
                    self.dic_datas[files_cubs[img_index]] = datas_
        skipped_sequences.close()
        print("Total image: ", self.im_nb)
        write_unseen_images(pres=self.p.PATH_TO_SKIPPED_SEQUENCES, substr='wall_', pimg=self.p.PATH_TO_SEQUENCES, keys=list(self.dic_datas.keys()))
        save_dic_to_HDF5(self.dic_datas, os.path.join(self.p.PATH_TO_SAVE_DATASET, self.p.DATABASE_NAME[0]+ "_wall.h5"))

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
                tmp_gr_mask.create_dataset(name=patch_id, data=data["patch_mask"][patch_id], dtype=np.uint8)                     # we convert in uint8 order to gain memory usage
                tmp_gr_img.create_dataset(name=patch_id, data=data["patch_Image_org"][patch_id], dtype=np.float32)
                tmp_gr_sr.create_dataset(name=patch_id, data=data["spatial_resolution"][patch_id], dtype=np.float32)

        f.close()