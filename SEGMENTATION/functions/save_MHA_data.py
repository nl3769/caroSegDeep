import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
from scipy.io import savemat

import vtk
from vtkmodules.util.numpy_support import *
from vtkmodules.vtkCommonCore import *
from vtkmodules.vtkIOImage import vtkMetaImageWriter
# ------------------------------------------------------------------------------------------------------------------------------------------------
def numpy_to_image(numpy_array, type):

    """
    @brief Convert a numpy 2D or 3D array to a vtkImageData object
    @param numpy_array 2D or 3D numpy array containing image data
    @return vtkImageData with the numpy_array content
    """

    shape = numpy_array.shape
    if len(shape) < 2:
        raise Exception('numpy array must have dimensionality of at least 2')

    h, w = shape[0], shape[1]
    c = 1

    if len(shape) == 3:
        c = shape[2]

    # Reshape 2D image to 1D array suitable for conversion to a
    # vtkArray with numpy_support.numpy_to_vtk()
    linear_array = np.reshape(numpy_array, (w*h, c))
    vtk_array = numpy_to_vtk(linear_array)

    image = vtk.vtkImageData()
    image.SetDimensions(w, h, 1)

    if type=='uint8':
        image.AllocateScalars(VTK_UNSIGNED_SHORT, 1)
    elif type=='float32':
        image.AllocateScalars(VTK_FLOAT, 1)

    image.GetPointData().GetScalars().DeepCopy(vtk_array)

    return image

# ------------------------------------------------------------------------------------------------------------------------------------------------
def saveMHA(pathToSave, VTK_image, info):

    VTK_image.SetSpacing(info['xSpacing'], info['ySpacing'], 1)
    VTK_image.SetOrigin(info['xOrigin'], info['yOrigin'], 1)

    w = vtkMetaImageWriter()
    w.SetInputData(VTK_image)
    w.SetCompression(True)
    # w.SetFileDimensionality(3)
    w.SetFileName(pathToSave)
    w.Write()

# ------------------------------------------------------------------------------------------------------------------------------------------------
def saveImage(path, image, type):
    path = path.split('/')
    patientName = path[-1].split('.')[0]
    dataBase = path[-2]

    pathToSave = '../results/AutomaticMethod/' + dataBase + '/' + type

    if os.path.exists(pathToSave) == False:
        os.makedirs(pathToSave)

    if len(image.shape) == 2:
        plt.imsave(pathToSave + '/' + patientName + '.png', image, cmap='gray')
    else:
        plt.imsave(pathToSave + '/' + patientName + '.png', image.astype(np.uint8))

# ------------------------------------------------------------------------------------------------------------------------------------------------
def saveMat(path, dic, type, interface=''):
    path = path.split('/')
    patientName = path[-1].split('.')[0]
    dataBase = path[-2]

    pathToSave = '../results/AutomaticMethod/' + dataBase + '/' + type

    if os.path.exists(pathToSave) == False:
        os.makedirs(pathToSave)

    savemat(pathToSave + '/' + patientName + interface + '.mat', dic)

# ------------------------------------------------------------------------------------------------------------------------------------------------
def createDirectory(path):
    try:
        os.makedirs(path)
    except OSError:
        print("The directory %s already exists." % path)
    else:
        print("Successfully created the directory %s " % path)

    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


# ------------------------------------------------------------------------------------------------------------------------------------------------
def saveOriginalImagesMHA(image, patient, seq, p):
    # path = "../../../../../../projects_results/carotid_segmentation_results/results_01_mha/caroSeg_MEIBURGER/laine/original_images/test_01/" + patient.split('.')[0] + '/' + "condition_01" + "/"
    path = os.path.join(p.PATH_TO_SAVE_EDUARDO_VIEWER, 'original_images/test_01', patient.split('.')[0], 'condition_01')
    createDirectory(path)
    spacing = seq.spacing

    info = {"xSpacing": spacing,
            "ySpacing": spacing,
            "xOrigin": 0,
            "yOrigin": 0}

    # --- prediction map intima media
    image = numpy_to_image(image, type='uint8')
    saveMHA(path + "/input_image.mha", image, info)

# ------------------------------------------------------------------------------------------------------------------------------------------------
def saveCaroSegDeepMHA(patch_img, patch_mask, distanceMap, postProcessing, pred_LI, pred_MA, pred_init, patient, seq, algoName, p):

    xSpacing = seq.spacing
    ySpacing = seq.spacing / seq.scale

    path = os.path.join(p.PATH_TO_SAVE_EDUARDO_VIEWER, algoName + '/test_01/' + patient.split('.')[0] + '/condition_01')
    createDirectory(path)

    info = {"xSpacing": xSpacing,
            "ySpacing": ySpacing,
            "xOrigin": 0,
            "yOrigin": 0}

    # --- prediction map intima media
    distanceMap = numpy_to_image(distanceMap, type='float32')
    saveMHA(path + "/prediction_intima_media.mha", distanceMap, info)

    # --- post processing intima media
    postProcessing = postProcessing.astype(np.uint8)
    distanceMap = numpy_to_image(postProcessing, 'uint8')
    saveMHA(path + "/post_processing_intima_media.mha", distanceMap, info)


    # --- patch
    id = 0

    for patch_id in range(len(patch_img)):
        patch = patch_img[patch_id]
        patchImg = patch['patch']
        patchMask = patch_mask[patch_id, :, :, 0]
        # diff = seq.annotationClass.borders['rightBorder'] - seq.annotationClass.borders['leftBorder']
        # if patch["(x, y)"][0] >= diff and patch["(x, y)"][0] <= 2*diff:
        if id < 10:
            pathPatchImg = path + "/patch_00" + str(id) + ".mha"
            pathPatchMask = path + "/patch_00" + str(id) + "_mask.mha"
        elif id < 100:
            pathPatchImg = path + "/patch_0" + str(id) + ".mha"
            pathPatchMask = path + "/patch_0" + str(id) + "_mask.mha"
        else:
            pathPatchImg = path + "/patch_" + str(id) + ".mha"
            pathPatchMask = path + "/patch_" + str(id) + "_mask.mha"

        info = {"xSpacing": xSpacing,
                "ySpacing": ySpacing,
                "xOrigin": patch["(x, y)"][0],
                "yOrigin": patch["(x, y)"][1]}

        # --- patch
        patchImg = numpy_to_image(patchImg, type='float32')
        saveMHA(pathPatchImg, patchImg, info)
        # --- prediction of the patch
        patchMask = numpy_to_image(patchMask, type='float32')
        saveMHA(pathPatchMask, patchMask, info)

        id += 1

    # --- prediction
    leftBorderROI = seq.annotationClass.bordersROI['leftBorder']
    rightBorderROI = seq.annotationClass.bordersROI['rightBorder']
    leftBorder = seq.annotationClass.borders['leftBorder']
    rightBorder = seq.annotationClass.borders['rightBorder']
    LIFile = open(os.path.join(path, "pred_LI.txt"), "w+")
    MAFile = open(os.path.join(path, "pred_MA.txt"), "w+")
    initFile = open(os.path.join(path, "init.txt"), "w+")

    for k in range(leftBorder, rightBorder):
        LIFile.write(str(k) + " " + str(pred_LI[k] / seq.scale) + "\n")
        MAFile.write(str(k) + " " + str(pred_MA[k] / seq.scale) + "\n")

    for k in range(leftBorderROI, rightBorderROI):
        initFile.write(str(k) + " " + str(pred_init[k] / seq.scale) + "\n")

    LIFile.close()
    MAFile.close()
    initFile.close()

# ------------------------------------------------------------------------------------------------------------------------------------------------
def saveExpert(patient, pathToAnnotation, p):

    expert_01 = "expert_01"
    expert_01bis = "expert_01bis"
    expert_02 = "expert_02"


    pathExpert_01 = os.path.join(p.PATH_TO_SAVE_EDUARDO_VIEWER, 'experts_points', expert_01, patient.split('.')[0] + "/condition_01/")
    pathExpert_01bis = os.path.join(p.PATH_TO_SAVE_EDUARDO_VIEWER, 'experts_points', expert_01bis,patient.split('.')[0] + "/condition_01/")
    pathExpert_02 = os.path.join(p.PATH_TO_SAVE_EDUARDO_VIEWER, 'experts_points', expert_02, patient.split('.')[0] + "/condition_01/")
    # pathExpert_01 = "../results/caroSeg_MEIBURGER/laine/experts_points/" + expert_01 + "/" + patient.split('.')[0] + "/condition_01/"
    # pathExpert_01 = os.path.join(pathtmp, expert_01, "condition_01")
    # pathExpert_01bis = os.path.join(pathtmp, expert_01bis, "condition_01")
    # pathExpert_02 = os.path.join(pathtmp, expert_02, "condition_01")

    createDirectory(pathExpert_01)
    createDirectory(pathExpert_01bis)
    createDirectory(pathExpert_02)

    # --- expert A1
    expert1_LI = os.path.join(pathToAnnotation, "../EXPERTS/GT-GV", patient.split('.')[0] + "-LI.txt")
    expert1_MA = os.path.join(pathToAnnotation, "../EXPERTS/GT-GV", patient.split('.')[0] + "-MA.txt")
    targetExpert1_LI = pathExpert_01 + "LI.txt"
    targetExpert1_MA = pathExpert_01 + "MA.txt"
    shutil.copyfile(expert1_LI, targetExpert1_LI)
    shutil.copyfile(expert1_MA, targetExpert1_MA)

    # --- expert A2
    expert2_LI = os.path.join(pathToAnnotation, "../EXPERTS/GT-GV", patient.split('.')[0] + "-LI.txt")
    expert2_MA = os.path.join(pathToAnnotation, "../EXPERTS/GT-GV", patient.split('.')[0] + "-MA.txt")
    targetExpert2_LI = pathExpert_01bis + "LI.txt"
    targetExpert2_MA = pathExpert_01bis + "MA.txt"
    shutil.copyfile(expert2_LI, targetExpert2_LI)
    shutil.copyfile(expert2_MA, targetExpert2_MA)

    # --- expert A3
    expert3_LI = os.path.join(pathToAnnotation, "../EXPERTS/GT-LG", patient.split('.')[0] + "-LI.txt")
    expert3_MA = os.path.join(pathToAnnotation, "../EXPERTS/GT-LG", patient.split('.')[0] + "-MA.txt")
    targetExpert3_LI = pathExpert_02 + "LI.txt"
    targetExpert3_MA = pathExpert_02 + "MA.txt"
    shutil.copyfile(expert3_LI, targetExpert3_LI)
    shutil.copyfile(expert3_MA, targetExpert3_MA)