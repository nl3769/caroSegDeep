import h5py

from tqdm import tqdm
import os
import numpy as np
from functions.load_datas import load_borders, load_tiff, load_annotation, get_files
from functions.patch_extraction import patch_extraction
from functions.split_data import split_data_fold
import matplotlib.pyplot as plt


class datasetBuilder():
    def __init__(self, p):

        self.window = p.PATCH_WIDTH
        self.overlay = p.PATCH_OVERLAY
        self.scale = p.SCALE
        self.dic_datas = {}
        self.imgNumber = 0
        self.p = p

    def build_data(self):
        skipped_sequences = open(os.path.join(self.p.PATH_TO_SKIPPED_SEQUENCES, "skipped_sequences.txt"), "w")
        for data_base in self.p.DATABASE_NAME:
            
            files_cubs = get_files(self.p.PATH_TO_SEQUENCES)

            for img_index in tqdm(range(len(files_cubs)), ascii=True, desc='Patch (Mask + image): ' + data_base):
                # --- load the image and the calibration factor
                spatial_res_y, img_f = load_tiff(os.path.join(self.p.PATH_TO_SEQUENCES, files_cubs[img_index]), PATH_TO_CF=self.p.PATH_TO_CF)
                spatial_resolution_x = spatial_res_y
                # --- load left and right borders
                borders = load_borders(os.path.join(self.p.PATH_TO_BORDERS, files_cubs[img_index].split('.')[0] + "_borders.mat"))
                # --- load the annotation
                mat = load_annotation(self.p.PATH_TO_CONTOUR, files_cubs[img_index], self.p.EXPERT)

                datas_, self.imgNumber = patch_extraction(img = img_f,
                                                          manualDel = mat,
                                                          borders = borders,
                                                          widthWindow = self.window,
                                                          overlay = self.overlay,
                                                          nameSeq = files_cubs[img_index],
                                                          resizeCol = self.scale,
                                                          skipped_sequences = skipped_sequences,
                                                          spatial_res_y = spatial_res_y,
                                                          spatial_res_x = spatial_resolution_x,
                                                          desiredSpatialResolution = self.p.SPATIAL_RESOLUTION,
                                                          imgNumber = self.imgNumber)

                if datas_ != "skipped":
                    self.dic_datas[files_cubs[img_index]] = datas_

        skipped_sequences.close()

        print("Total image: ", self.imgNumber)

        data = self.dic_datas.copy()
        unseen_images = open(os.path.join(self.p.PATH_TO_SKIPPED_SEQUENCES, "unseen_images.txt"), "w")
        self.dic_datas = split_data_fold(data=data.copy(),
                                         patientList=self.dic_datas.keys(),
                                         file=unseen_images,
                                         p=self.p)
        self.save_dic_to_HDF5(path=os.path.join(self.p.PATH_TO_SAVE_DATASET, self.p.DATABASE_NAME[0]+ ".h5"))

        unseen_images.close()
    # ----------------------------------------------------------------
    def save_dic_to_HDF5(self, path):
        f = h5py.File(path, "w")
        for key in self.dic_datas.keys():
            print(key)

            gr = key
            tmpGr = f.create_group(name=gr)

            tmpGrMask = tmpGr.create_group(name="masks")
            tmpGrImg = tmpGr.create_group(name="img")
            tmpGrPatientName = tmpGr.create_group(name="patientName")
            tmpGrSR = tmpGr.create_group(name="spatial_resolution")

            data = self.dic_datas[key]

            for i in data["masks"].keys():
                tmpGrMask.create_dataset(name=i, data=data["masks"][i], dtype=np.uint8)
                tmpGrImg.create_dataset(name=i, data=data["images"][i], dtype=np.float32)
                tmpGrSR.create_dataset(name=i, data=data["spatial_resolution"][i], dtype=np.float32)

            for j in range(len(data["patient_name"])):
                tmpGrPatientName.create_dataset(name="Patient_" + str(j) + "_name", data=data["patient_name"][j])

        f.close()
