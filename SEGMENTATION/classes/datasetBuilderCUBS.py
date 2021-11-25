'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import h5py

from tqdm import tqdm
import os
import numpy as np
from functions.load_datas import load_borders, load_tiff, load_annotation, get_files
from functions.patch_extraction import patch_extraction
from functions.split_data import split_data_fold

class datasetBuilder():
    def __init__(self, p):
        ''' class to compute dataset according to CUBS database '''
        self.window = p.PATCH_WIDTH
        self.overlay = p.PATCH_OVERLAY
        self.scale = p.SCALE
        self.dic_datas = {}
        self.im_nb = 0
        self.p = p
    # ------------------------------------------------------------------------------------------------------------------
    def build_data(self):
        ''' build_data compute and write the dataset in .h5 file. The h5 contains training, validation and validation set. '''
        skipped_sequences=open(os.path.join(self.p.PATH_TO_SKIPPED_SEQUENCES, "skipped_sequences.txt"), "w") # contains images that cannot be incorporated into the data set
        # --- loop is required if more than one database is used. It not necessary for CUBS
        for data_base in self.p.DATABASE_NAME:
            files_cubs = get_files(self.p.PATH_TO_SEQUENCES) # name of all images in self.p.PATH_TO_SEQUENCES
            # --- loop over all the images in the database
            # for img_index in tqdm(range(len(files_cubs)), ascii=True, desc='Patch (Mask + image): ' + data_base):
            for img_index in tqdm(range(len(files_cubs[0:100])), ascii=True, desc='Patch (Mask + image): ' + data_base):
                # --- load the image and the calibration factor
                spatial_res_y, img_f = load_tiff(os.path.join(self.p.PATH_TO_SEQUENCES, files_cubs[img_index]), PATH_TO_CF=self.p.PATH_TO_CF)
                spatial_resolution_x = spatial_res_y
                # --- load left and right borders
                borders = load_borders(os.path.join(self.p.PATH_TO_BORDERS, files_cubs[img_index].split('.')[0] + "_borders.mat"))
                # --- load annotation
                mat = load_annotation(self.p.PATH_TO_CONTOUR, files_cubs[img_index], self.p.EXPERT)
                # --- extracts patch for the current image
                datas_, self.im_nb = patch_extraction(img=img_f,
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
                # --- we add to self.dic_datas if patches were extracted
                if datas_ != "skipped":
                    self.dic_datas[files_cubs[img_index]] = datas_
        skipped_sequences.close()
        print("Total image: ", self.im_nb)
        data = self.dic_datas.copy()
        unseen_images = open(os.path.join(self.p.PATH_TO_SKIPPED_SEQUENCES, "unseen_images.txt"), "w")
        self.dic_datas = split_data_fold(data=data,
                                         file=unseen_images,
                                         p=self.p)
        self.save_dic_to_HDF5(path=os.path.join(self.p.PATH_TO_SAVE_DATASET, self.p.DATABASE_NAME[0]+ ".h5"))

        unseen_images.close()
    # ------------------------------------------------------------------------------------------------------------------
    def save_dic_to_HDF5(self, path):
        ''' Saves patches in .h5 file with train/validation/test sets. '''
        f = h5py.File(path, "w")
        for key in self.dic_datas.keys():
            print(f"The {key} set is being written.")

            gr = key
            tmpGr = f.create_group(name=gr)

            tmp_gr_mask = tmpGr.create_group(name="masks")                      # groupe for mask
            tmp_gr_img = tmpGr.create_group(name="img")                         # groupe for patch (img)
            tmp_gr_patient_name = tmpGr.create_group(name="patientName")        # groupe for name
            tmp_gr_sr = tmpGr.create_group(name="spatial_resolution")           # groupe for spatial resolution (used for hausdorff distance)

            data = self.dic_datas[key]

            # --- Writes information
            for i in data["masks"].keys():
                tmp_gr_mask.create_dataset(name=i, data=data["masks"][i], dtype=np.uint8)       # we convert in uint8 order to gain memory usage
                tmp_gr_img.create_dataset(name=i, data=data["images"][i], dtype=np.float32)
                tmp_gr_sr.create_dataset(name=i, data=data["spatial_resolution"][i], dtype=np.float32)

            for j in range(len(data["patient_name"])):
                tmp_gr_patient_name.create_dataset(name="Patient_" + str(j) + "_name", data=data["patient_name"][j])

        f.close()
