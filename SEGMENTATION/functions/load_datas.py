import numpy as np
from skimage.color import rgb2gray
from pydicom import dcmread
import scipy.io
import matplotlib.pyplot as plt
import cv2
import os

def load_data(sequence,
            spatialResolution,
            fullSequence,
            p):

    extension = sequence.split('.')
    LOI = False
    if extension[-1] != "mat" and extension[-1] != "tiff":
        extension = 'dicom'
    else:
        extension = extension[-1]

    if sequence.split('/')[-1].split('_')[0] == 'LOI':
        LOI =  True



    if extension == 'dicom':
        return load_DICOM_sequence(sequence=sequence,
                                 spatialResolution=spatialResolution,
                                 fullSequence=fullSequence)
    if extension == 'mat' and LOI == False:
        return load_MAT_sequence(sequence=sequence,
                               spatialResolution=spatialResolution,
                               fullSequence=fullSequence)

    if extension == 'mat' and LOI == True:
        return load_MAT_image(sequence=sequence,
                               spatialResolution=spatialResolution)

    if extension == 'tiff':
        return load_TIFF_image(sequence=sequence,
                             spatialResolution=spatialResolution,
                             CFPath=p.PATH_TO_CF)
# ----------------------------------------------------------------
def load_DICOM_sequence(sequence,
                      spatialResolution,
                      fullSequence):

    sq = dcmread(sequence)
    spatialResolutionY = sq.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
    sq = sq.pixel_array
    dimension = sq.shape  # get the dimension of the sequence
    firstFrame = np.uint8(rgb2gray(sq[0,]) * 255)

    if fullSequence == True:
        dimSq = sq.shape
        out = np.zeros((dimSq[0],) + (dimSq[1],) + (dimSq[2],))

        # get the first image of the sequence
        for i in range(dimSq[0]):
            out[i, :, :] = np.float32(rgb2gray(sq[i,]) * 255)

        out, scale = sequence_preprocessing(sequence = out,
                                           currentSpatialResolution = spatialResolutionY,
                                           spatialResolution = spatialResolution)

    elif fullSequence == False:
        dimSq = sq.shape
        out = np.zeros((1,) + (dimSq[1],) + (dimSq[2],))

        # --- get the first image of the sequence
        out[0,] = np.float32(rgb2gray(sq[0,]) * 255)
        out, scale = sequence_preprocessing(sequence=out,
                                           currentSpatialResolution=spatialResolutionY,
                                           spatialResolution=spatialResolution)

    return out, scale, spatialResolutionY, firstFrame
# ----------------------------------------------------------------
def load_MAT_sequence(sequence,
                    spatialResolution,
                    fullSequence):
    data = scipy.io.loadmat(sequence)
    seq = data['ImgTmp']
    seq = np.moveaxis(seq, -1, 0)
    spatialResolutionY = 0.0034
    firstFrame = seq[0,:,:].copy()

    if fullSequence == True:
        out, scale = sequence_preprocessing(sequence = seq,
                                           currentSpatialResolution = spatialResolutionY,
                                           spatialResolution = spatialResolution)

    elif fullSequence == False:
        out, scale = sequence_preprocessing(sequence=np.expand_dims(seq[0,], axis=0),
                                           currentSpatialResolution=spatialResolutionY,
                                           spatialResolution=spatialResolution)

    return out, scale, spatialResolutionY, firstFrame
# ----------------------------------------------------------------
def load_MAT_image(sequence,
                 spatialResolution):
    data = scipy.io.loadmat(sequence)
    seq = data['ima']
    spatialResolutionY = 0.0067
    firstFrame = seq.copy()

    out, scale = sequence_preprocessing(sequence=np.expand_dims(seq, axis=0),
                                       currentSpatialResolution=spatialResolutionY,
                                       spatialResolution=spatialResolution)

    return out, scale, spatialResolutionY, firstFrame
# ----------------------------------------------------------------
def load_TIFF_image(sequence,
                   spatialResolution,
                   CFPath):
    seq = plt.imread(sequence)
    if len(seq.shape) == 3:
        seq = seq[:,:,0]
    pathToSpacing = os.path.join(CFPath, sequence.split('/')[-1].split('.')[0] + "_CF.txt")
    spatialResolutionY = read_CF_directory(pathToSpacing)

    firstFrame = seq.copy()

    out, scale = sequence_preprocessing(sequence=np.expand_dims(seq, axis=0),
                                       currentSpatialResolution=spatialResolutionY,
                                       spatialResolution=spatialResolution)

    return out, scale, spatialResolutionY, firstFrame
# ----------------------------------------------------------------
def read_CF_directory(path):
    f = open(path, "r")
    val = f.readline().split(' \n')
    return float(val[0])/10
# ----------------------------------------------------------------
def sequence_preprocessing(sequence,
                          currentSpatialResolution,
                          spatialResolution):

    dimFrame = sequence.shape
    heightFrame = dimFrame[1]
    scale = (currentSpatialResolution * 10000) / spatialResolution
    height_tmp = round(scale * heightFrame)

    scale = height_tmp / heightFrame

    out = np.zeros((dimFrame[0],) + (height_tmp,) + (dimFrame[2],))

    for i in range(dimFrame[0]):
        out[i, :, :] = cv2.resize(sequence[i, :, :].astype(np.float32), (dimFrame[2], height_tmp), interpolation=cv2.INTER_LINEAR)


    return out.astype(np.float32), scale
# ----------------------------------------------------------------
def load_borders(path):
    mat_b = scipy.io.loadmat(path)
    right_b = mat_b['border_right']
    right_b = right_b[0, 0]
    left_b = mat_b['border_left']
    left_b = left_b[0, 0]

    return left_b-1, right_b-1
# ----------------------------------------------------------------
def load_tiff(sequence,
             PATH_TO_CF):
    seq = plt.imread(sequence)

    if len(seq.shape) == 3:
        seq = seq[:, :, 0]

    pathToSpacing = os.path.join(PATH_TO_CF, sequence.split('/')[-1].split('.')[0] + "_CF.txt")
    spatialResolutionY = read_CF_file(pathToSpacing)

    return spatialResolutionY, seq
# ----------------------------------------------------------------
def read_CF_file(path):
    f = open(path, "r")
    val = f.readline().split(' \n')
    return float(val[0])/10
# ----------------------------------------------------------------
def load_annotation(path,
                   patient,
                   nameExpert):


    dataPathIFC3 = path + '/' + patient.split('.')[0] + "_IFC3_" + nameExpert + ".mat"
    dataPathIFC4 = path + '/' + patient.split('.')[0] + "_IFC4_" + nameExpert + ".mat"

    mat_IFC3 = scipy.io.loadmat(dataPathIFC3)
    mat_IFC3 = mat_IFC3['seg']

    mat_IFC4 = scipy.io.loadmat(dataPathIFC4)
    mat_IFC4 = mat_IFC4['seg']

    return mat_IFC3, mat_IFC4
# ----------------------------------------------------------------
def get_files(path):
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