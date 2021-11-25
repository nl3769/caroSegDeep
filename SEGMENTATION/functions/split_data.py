'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import _io

def split_data_fold(data: dict, file: _io.TextIOWrapper, p):

    ''' Create training/validation/testing set according to .txt files. '''

    dataset = {"train":      {"masks": {}, "images": {}, "patient_name": [], 'spatial_resolution': {}},
               "validation": {"masks": {}, "images": {}, "patient_name": [], 'spatial_resolution': {}},
               "test":       {"masks": {}, "images": {}, "patient_name": [], 'spatial_resolution': {}}}

    sets = os.listdir(p.PATH_TO_FOLDS)

    var=0

    for set in sets:
        print(f"set: {set}")
        # --- we first read the .txt file which contains the sets
        with open(os.path.join(p.PATH_TO_FOLDS, set)) as f:
            patients = f.readlines()

        for patient in range(len(patients)):
            patients[patient]=patients[patient].split('.')[0]

        if "TrainList" in set:
            key = "train"
        if "TestList" in set:
            key = "test"
        if "ValList" in set:
            key = "validation"


        for name in patients:
            name = name + ".tiff"
            var = var+1
            print(f"var {var}")
            if name in data.keys():

                masks_ = data[name]["patch_mask"]
                images_ = data[name]["patch_Image_org"]
                spatial_res_ = data[name]["spatial_resolution"]

                dataset[key]["patient_name"].append(name)
                for img in masks_.keys():
                    dataset[key]["masks"][img] = masks_[img]
                    dataset[key]["images"][img] = images_[img]
                    dataset[key]["spatial_resolution"][img] = spatial_res_[img]

                del data[name]
            else:
                file.write(name + "\n")

    return dataset