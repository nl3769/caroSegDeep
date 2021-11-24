import os

def split_data_fold(data, patientList, file, p):

    dataset = {"train":      {"masks": {}, "images": {}, "patient_name": [], 'spatial_resolution': {}},
               "validation": {"masks": {}, "images": {}, "patient_name": [], 'spatial_resolution': {}},
               "test":       {"masks": {}, "images": {}, "patient_name": [], 'spatial_resolution': {}}}

    sets = os.listdir(p.PATH_TO_FOLDS)

    for set in sets:
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

            if name in data.keys():

                tmpmasks = data[name]["patch_mask"]
                tmpimages = data[name]["patch_Image_org"]
                tmpspatialResolution = data[name]["spatial_resolution"]

                dataset[key]["patient_name"].append(name)
                for img in tmpmasks.keys():
                    dataset[key]["masks"][img] = tmpmasks[img]
                    dataset[key]["images"][img] = tmpimages[img]
                    dataset[key]["spatial_resolution"][img] = tmpspatialResolution[img]

                del data[name]
            else:
                file.write(name + "\n")

    return dataset