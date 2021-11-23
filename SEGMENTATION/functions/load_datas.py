import numpy as np
from skimage.color import rgb2gray
from pydicom import dcmread
import scipy.io
import matplotlib.pyplot as plt
import cv2
import os

def LoadData(sequence,
            spatialResolution,
            fullSequence,
            p):

    extension = sequence.split('.')
    LOI = False
    if extension[-1] != "mat" and extension[-1] != "tiff":
        extension0 = 'dicom'
    else:
        extension0 = extension[-1]

    if sequence.split('/')[-1].split('_')[0] == 'LOI':
        LOI =  True



    if extension0 == 'dicom':
        return LoadDICOMSequence(sequence=sequence,
                                 spatialResolution=spatialResolution,
                                 fullSequence=fullSequence)
    if extension0 == 'mat' and LOI == False:
        return LoadMATSequence(sequence=sequence,
                               spatialResolution=spatialResolution,
                               fullSequence=fullSequence)

    if extension0 == 'mat' and LOI == True:
        return LoadMATImage(sequence=sequence,
                               spatialResolution=spatialResolution)

    if extension0 == 'tiff':
        return LoadTiffImage(sequence=sequence,
                             spatialResolution=spatialResolution,
                             CFPath=p.PATH_TO_CF)

def LoadFirstFrameFromDICOM(sequences_path,
                            name_database,
                            files_sequences_f,
                            img_index):

    sq = dcmread(sequences_path + name_database + files_sequences_f[img_index])
    spatialResolutionY = sq.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
    spatialResolutionX = sq.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    sq = sq.pixel_array
    dimension = sq.shape  # get the dimension of the sequence
    img_nb = dimension[0]  # get the number of sequences

    # get the first image of the sequence
    current_img = sq[0, :, :, :]
    current_img = np.uint8(rgb2gray(current_img) * 255)

    return current_img, spatialResolutionX, spatialResolutionY

def LoadFirstFrameMat(sequences_path,
                      name_database,
                      patientName):

    img_t = scipy.io.loadmat(sequences_path + name_database + patientName)
    img_t = img_t['ima']

    return img_t

def LoadBorders(path):
    mat_b = scipy.io.loadmat(path)
    right_b = mat_b['border_right']
    right_b = right_b[0, 0]
    left_b = mat_b['border_left']
    left_b = left_b[0, 0]

    return left_b-1, right_b-1

def LoadBordersUsingName(borders_path,
                         name_database,
                         borders_name):

    mat_b = scipy.io.loadmat(borders_path + name_database + borders_name)
    right_b = mat_b['border_right']
    right_b = right_b[0, 0]
    left_b = mat_b['border_left']
    left_b = left_b[0, 0]

    return left_b, right_b

def LoadAnnotation(path,
                   databaseName,
                   patient,
                   nameExpert):


    dataPathIFC3 = path + databaseName + '/' + patient.split('.')[0] + "_IFC3_" + nameExpert + ".mat"
    dataPathIFC4 = path + databaseName + '/' + patient.split('.')[0] + "_IFC4_" + nameExpert + ".mat"

    mat_IFC3 = scipy.io.loadmat(dataPathIFC3)
    mat_IFC3 = mat_IFC3['seg']

    mat_IFC4 = scipy.io.loadmat(dataPathIFC4)
    mat_IFC4 = mat_IFC4['seg']

    return mat_IFC3, mat_IFC4

def LoadAnnotationUsingName(contours_path,
                            name_database,
                            nameAnnotationIFC3,
                            nameAnnotationIFC4):

    mat_IFC3 = scipy.io.loadmat(contours_path + name_database + nameAnnotationIFC3)
    mat_IFC4 = scipy.io.loadmat(contours_path + name_database + nameAnnotationIFC4)

    return {"IFC3": mat_IFC3['seg'][0, :], "IFC4": mat_IFC4['seg'][0, :] }

def LoadDICOMSequence(sequence,
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

        out, scale = SequencePreProcessing(sequence = out,
                                           currentSpatialResolution = spatialResolutionY,
                                           spatialResolution = spatialResolution)

    elif fullSequence == False:
        dimSq = sq.shape
        out = np.zeros((1,) + (dimSq[1],) + (dimSq[2],))

        # get the first image of the sequence

        out[0,] = np.float32(rgb2gray(sq[0,]) * 255)

        out, scale = SequencePreProcessing(sequence=out,
                                           currentSpatialResolution=spatialResolutionY,
                                           spatialResolution=spatialResolution)

    return out, scale, spatialResolutionY, firstFrame

def LoadMATSequence(sequence,
                    spatialResolution,
                    fullSequence):
    data = scipy.io.loadmat(sequence)
    seq = data['ImgTmp']
    seq = np.moveaxis(seq, -1, 0)
    # spatialResolutionY = data['PhysicalDeltaY'][0][0]
    spatialResolutionY = 0.0034
    firstFrame = seq[0,:,:].copy()

    if fullSequence == True:
        dimSq = seq.shape
        out = np.zeros(dimSq)

        out, scale = SequencePreProcessing(sequence = seq,
                                           currentSpatialResolution = spatialResolutionY,
                                           spatialResolution = spatialResolution)

    elif fullSequence == False:

        out, scale = SequencePreProcessing(sequence=np.expand_dims(seq[0,], axis=0),
                                           currentSpatialResolution=spatialResolutionY,
                                           spatialResolution=spatialResolution)

    return out, scale, spatialResolutionY, firstFrame

def LoadMATImage(sequence,
                 spatialResolution):
    data = scipy.io.loadmat(sequence)
    seq = data['ima']
    spatialResolutionY = 0.0067
    firstFrame = seq.copy()

    out, scale = SequencePreProcessing(sequence=np.expand_dims(seq, axis=0),
                                       currentSpatialResolution=spatialResolutionY,
                                       spatialResolution=spatialResolution)

    return out, scale, spatialResolutionY, firstFrame

def LoadTiffImage(sequence,
                  spatialResolution,
                  CFPath):
    seq = plt.imread(sequence)

    if len(seq.shape) == 3:
        seq = seq[:,:,0]

    pathToSpacing = os.path.join(CFPath, sequence.split('/')[-1].split('.')[0] + "_CF.txt")
    spatialResolutionY = readCFDirectory(pathToSpacing)

    firstFrame = seq.copy()

    out, scale = SequencePreProcessing(sequence=np.expand_dims(seq, axis=0),
                                       currentSpatialResolution=spatialResolutionY,
                                       spatialResolution=spatialResolution)

    return out, scale, spatialResolutionY, firstFrame

def LoadTiff(sequence):
    seq = plt.imread(sequence)

    if len(seq.shape) == 3:
        seq = seq[:, :, 0]

    pathToSpacing = "../../../data/CF/" + sequence.split('/')[-1].split('.')[0] + "_CF.txt"
    spatialResolutionY = readCFDirectory(pathToSpacing)

    return spatialResolutionY, seq

def readCFDirectory(path):
    f = open(path, "r")
    val = f.readline().split(' \n')
    return float(val[0])/10

def SequencePreProcessing(sequence,
                          currentSpatialResolution,
                          spatialResolution):

    dimFrame = sequence.shape
    heightFrame = dimFrame[1]
    scale = (currentSpatialResolution * 10000) / spatialResolution
    height_tmp = round(scale * heightFrame)

    scale = height_tmp / heightFrame

    # LI = scale * LI
    # MA = scale * MA

    out = np.zeros((dimFrame[0],) + (height_tmp,) + (dimFrame[2],))

    for i in range(dimFrame[0]):
        out[i, :, :] = cv2.resize(sequence[i, :, :].astype(np.float32), (dimFrame[2], height_tmp), interpolation=cv2.INTER_LINEAR)


    return out.astype(np.float32), scale

def LoadSpatialResolution(sequence):

    sq = dcmread(sequence)
    spatialResolutionY = sq.SequenceOfUltrasoundRegions[0].PhysicalDeltaY

    return spatialResolutionY