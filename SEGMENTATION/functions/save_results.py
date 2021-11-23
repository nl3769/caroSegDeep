import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

def saveImage(path, image, type):

    path = path.split('/')
    patientName = path[-1].split('.')[0]
    dataBase = path[-2]

    pathToSave = 'results/meiburger/AutomaticMethod/' + dataBase + '/' + type

    if os.path.exists(pathToSave)==False:
        os.makedirs(pathToSave)

    if len(image.shape) == 2:
        plt.imsave(pathToSave+ '/' + patientName + '.png', image, cmap='gray')
    else:
        plt.imsave(pathToSave + '/' + patientName + '.png', image.astype(np.uint8))

def saveMat(path, dic, type, interface=''):

    path = path.split('/')
    patientName = path[-1].split('.')[0]
    dataBase = path[-2]

    pathToSave = 'results/meiburger/AutomaticMethod/' + dataBase + '/' + type

    if os.path.exists(pathToSave)==False:
        os.makedirs(pathToSave)

    savemat(pathToSave + '/' + patientName + interface + '.mat', dic)
