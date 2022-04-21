'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import numpy as np
from skimage.color import rgb2gray
from pydicom import dcmread
import scipy.io
import matplotlib.pyplot as plt
import cv2
import os

def load_data(sequence: str,
              spatial_res: int,
              full_seq: bool,
              p):
    ''' Loads any type of date (mat, dicom, tiff) and returns the interpolated image and the scale coefficient. '''
    extension = sequence.split('.')
    LOI = False
    if extension[-1] != "mat" and extension[-1] != "tiff":
        extension = 'dicom'
    else:
        extension = extension[-1]

    if sequence.split('/')[-1].split('_')[0] == 'LOI':
        LOI =  True
    if extension == 'dicom':
        return load_DICOM_sequence(sequence=sequence, spatial_res=spatial_res, full_seq=full_seq)
    if extension == 'mat' and LOI == False:
        return load_MAT_sequence(sequence=sequence, spatial_res=spatial_res, full_seq=full_seq)
    if extension == 'mat' and LOI == True:
        return load_MAT_image(sequence=sequence, spatial_res=spatial_res)
    if extension == 'tiff':
        return load_TIFF_image(sequence=sequence, spatial_res=spatial_res, CF_path=p.PATH_TO_CF)
# ----------------------------------------------------------------
def load_DICOM_sequence(sequence: str, spatial_res: int, full_seq: bool):
    ''' Loads the first frame or the sequence stored in DICOM and returns the interpolated image(s) and the scale coefficient. '''
    sq = dcmread(sequence)
    spatial_res_y = sq.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
    sq = sq.pixel_array
    firstFrame = np.uint8(rgb2gray(sq[0,]) * 255)

    if full_seq == True:
        dimSq = sq.shape
        out = np.zeros((dimSq[0],) + (dimSq[1],) + (dimSq[2],))

        # get the first image of the sequence
        for i in range(dimSq[0]):
            out[i, :, :] = np.float32(rgb2gray(sq[i,]) * 255)

        out, scale = sequence_preprocessing(sequence = out, current_spatial_res = spatial_res_y, spatial_res = spatial_res)

    elif full_seq == False:
        dimSq = sq.shape
        out = np.zeros((1,) + (dimSq[1],) + (dimSq[2],))

        # --- get the first image of the sequence
        out[0,] = np.float32(rgb2gray(sq[0,]) * 255)
        out, scale = sequence_preprocessing(sequence=out, current_spatial_res=spatial_res_y, spatial_res=spatial_res)

    return out, scale, spatial_res_y, firstFrame
  
# ----------------------------------------------------------------
def load_MAT_sequence(sequence: str, spatial_res: int, full_seq: bool):
    ''' Loads the first frame or the sequence stored in MAT and returns the interpolated image(s) and the scale coefficient. '''
    data = scipy.io.loadmat(sequence)
    seq = data['ImgTmp']
    seq = np.moveaxis(seq, -1, 0)
    spatial_res_y = 0.0034
    firstFrame = seq[0,:,:].copy()

    if full_seq == True:
        out, scale = sequence_preprocessing(sequence = seq,
                                           current_spatial_res = spatial_res_y,
                                           spatial_res = spatial_res)

    elif full_seq == False:
        out, scale = sequence_preprocessing(sequence=np.expand_dims(seq[0,], axis=0),
                                           current_spatial_res=spatial_res_y,
                                           spatial_res=spatial_res)

    return out, scale, spatial_res_y, firstFrame
# ----------------------------------------------------------------
def load_MAT_image(sequence: str, spatial_res: int):
    ''' Loads the first frame of a sequence stored in MAT and returns the interpolated image and the scale coefficient. '''
    data = scipy.io.loadmat(sequence)
    seq = data['ima']
    spatial_res_y = 0.0067
    firstFrame = seq.copy()

    out, scale = sequence_preprocessing(sequence=np.expand_dims(seq, axis=0),
                                       current_spatial_res=spatial_res_y,
                                       spatial_res=spatial_res)

    return out, scale, spatial_res_y, firstFrame
  
# ----------------------------------------------------------------
def load_TIFF_image(sequence: str, spatial_res: int, CF_path: str):
    ''' Load tiff image and returns the interpolated image and the scale coefficient. '''
    seq = plt.imread(sequence)
    if len(seq.shape) == 3:
        seq = seq[:,:,0]
    path_spacing = os.path.join(CF_path, sequence.split('/')[-1].split('.')[0] + "_CF.txt")
    spatial_res_y = read_CF_directory(path_spacing)

    firstFrame = seq.copy()

    out, scale = sequence_preprocessing(sequence=np.expand_dims(seq, axis=0),
                                       current_spatial_res=spatial_res_y,
                                       spatial_res=spatial_res)

    return out, scale, spatial_res_y, firstFrame
# ----------------------------------------------------------------
def read_CF_directory(path: str):
    ''' Read calibration factor. '''
    f = open(path, "r")
    val = f.readline().split(' \n')
    return float(val[0])/10
  
# ----------------------------------------------------------------
def sequence_preprocessing(sequence: str, current_spatial_res: float, spatial_res: float):
    ''' Sequence preprocessing -> vertical interpolation to reach a vertical pixel size of spatial_resolution. '''
    dimFrame = sequence.shape
    heightFrame = dimFrame[1]
    scale = (current_spatial_res * 10000) / spatial_res
    height_tmp = round(scale * heightFrame)

    scale = height_tmp / heightFrame

    out = np.zeros((dimFrame[0],) + (height_tmp,) + (dimFrame[2],))

    for i in range(dimFrame[0]):
        out[i, :, :] = cv2.resize(sequence[i, :, :].astype(np.float32), (dimFrame[2], height_tmp), interpolation=cv2.INTER_LINEAR)


    return out.astype(np.float32), scale
  
# ----------------------------------------------------------------
def load_borders(path: str):
    ''' Loads borders in .mat file and return the left and the right borders. '''
    mat_b = scipy.io.loadmat(path)
    right_b = mat_b['border_right']
    right_b = right_b[0, 0]
    left_b = mat_b['border_left']
    left_b = left_b[0, 0]

    return left_b-1, right_b-1
  
# ----------------------------------------------------------------
def load_tiff(sequence: str, PATH_TO_CF: str):
    '''Loads a tiff image. For the CUBS database, each .tiff has an external .txt file that contains the pixel size (CF).
    Original image and the pixel size are returned.'''
    seq = plt.imread(sequence)
    if len(seq.shape) == 3:
        seq = seq[:, :, 0]
    path_spacing = os.path.join(PATH_TO_CF, sequence.split('/')[-1].split('.')[0] + "_CF.txt")
    spatial_res_y = read_CF_file(path_spacing)

    return spatial_res_y, seq
  
# ----------------------------------------------------------------
def read_CF_file(path: str):
    ''' Loads pixel size in .txt file for CUBS database. '''
    f = open(path, "r")
    val = f.readline().split(' \n')
    return float(val[0])/10
  
# ----------------------------------------------------------------
def load_annotation(path: str, patient: str, nameExpert: str):
    ''' Loads annotation stored in .mat file. The function returns two vectors corresponding to LI and MA interfaces.'''
    path_IFC3 = path + '/' + patient.split('.')[0] + "_IFC3_" + nameExpert + ".mat"
    path_IFC4 = path + '/' + patient.split('.')[0] + "_IFC4_" + nameExpert + ".mat"

    IFC3 = scipy.io.loadmat(path_IFC3)
    IFC3 = IFC3['seg']
    IFC4 = scipy.io.loadmat(path_IFC4)
    IFC4 = IFC4['seg']

    return IFC3, IFC4
  
# ----------------------------------------------------------------
def get_files(path: str):
    """ Returns a list containing the name of all the files. """
    file = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                file.append(entry.name)
    return file
