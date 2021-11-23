import scipy.io
import numpy as np
import math
from functions.processing import getBiggestConnexeRegion
# from classes.getFarWall import MousePts
import matplotlib.pyplot as plt
import cv2
class annotationClass():

    def __init__(self,
                 annotationPath,
                 nameAnnotation,
                 dimension,
                 borderPath,
                 firstFrame,
                 scale):

        self.borders = LoadBorders(borderPath)
        self.bordersROI = self.borders

        #### Remove later ####
        # keys = list(self.borders.keys())
        # print(keys[0] + " = ", self.borders[keys[0]])
        # print(keys[1] + " = ", self.borders[keys[1]])

        # print("Enter a new value, the difference between them has to be greater or equal to 128.")
        #
        # print(keys[0] + ":")
        # self.borders[keys[0]] = int(input())
        # print(keys[1] + ":")
        # self.borders[keys[1]] = int(input())
        # ######################

        self.mapAnnotation = np.zeros((dimension[0]+1, dimension[2], 2))
        self.sequenceDimension = dimension

        # self.annotation = self.initialization(contours_path = annotationPath,
        #                                       localization = pos,
        #                                       scale=scale)

        self.annotation = 0

    def yPosition(self,
                  xLeft,
                  width,
                  height,
                  frameID):

        xRight = xLeft + width

        # we load the position of the LI and MA interfaces
        posLI = self.mapAnnotation[frameID, :, 0][xLeft:xRight]
        posMA = self.mapAnnotation[frameID, :, 1][xLeft:xRight]

        # posLI = LItmp[xLeft:xRight]
        # posMA = MAtmp[xLeft:xRight]


        # we compute the mean value and retrieve the half height of a patch
        concatenation = np.concatenate((posMA, posLI))
        y = round(np.mean(concatenation) - height/2)

        # we just check if the value is greater than zero to avoid any problem
        if y < 0:
            print("Problem in function yPosition !!!!!!")
            y = 0

        return y


    def initialization(self,
                       contours_path,
                       localization,
                       scale):



        IFC3 = np.zeros(self.sequenceDimension[2])
        IFC4 = np.zeros(self.sequenceDimension[2])

        IFC3[self.borders['leftBorder']:self.borders['rightBorder']] = localization*scale
        IFC4[self.borders['leftBorder']:self.borders['rightBorder']] = localization*scale

        self.mapAnnotation[0, :, 0] = IFC3
        self.mapAnnotation[0, :, 1] = IFC4

        # # we remove the annotation which are out of the borders
        # self.mapAnnotation[0, 0:self.borders['leftBorder'], 0] = 0
        # self.mapAnnotation[0, self.borders['rightBorder']+1:, 0] = 0
        # self.mapAnnotation[0, 0:self.borders['leftBorder'], 1] = 0
        # self.mapAnnotation[0, self.borders['rightBorder']+1: , 1] = 0

        return {"IFC3": self.mapAnnotation[0, :, 0], "IFC4": self.mapAnnotation[0, :, 1]}

    def updateAnnotation(self,
                         previousMask,
                         frameID):

        # --- window of +/- neighbours pixels where the algorithm search the borders
        neighours = 100

        # --- the algorithm starts from the left to the right
        xStart = self.borders['leftBorder']
        xEnd = self.borders['rightBorder']

        # --- dimension of the mask
        dim = previousMask.shape
        # --- we extract the biggest connexe region
        previousMask = getBiggestConnexeRegion(previousMask)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(previousMask == 1))
        seed = (round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))
        # --- random value to j but not to high. It the first y coordinate at the xStart position
        j = 300
        # --- j cannot exceed limit
        limit = dim[0]-1
        # --- delimitation of the LI boundary
        for i in range(seed[1]+1, xEnd+1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previousMask[j, i] == 1):
                    self.mapAnnotation[frameID, i, 0] = j
                    # previousMask[j, i] = 100 # for debug
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    self.mapAnnotation[frameID, i, 0] = self.mapAnnotation[frameID, i-1, 0]
                    condition = False
                    # previousMask[j, i] = 100 # for debug

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2*neighours
        j = 300
        for i in range(seed[1], xStart-1, -1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previousMask[j, i] == 1):
                    self.mapAnnotation[frameID, i, 0] = j
                    # previousMask[j, i] = 100 # for debug
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    self.mapAnnotation[frameID, i, 0] = self.mapAnnotation[frameID, i-1, 0]
                    condition = False
                    # previousMask[j, i] = 100 # for debug

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2*neighours

        # --- delimitation of the MA boundary
        j = 300
        limit = dim[0] - 1
        for i in range(seed[1]+1, xEnd+1):
            condition = True

            while condition == True:

                if (j < dim[0] and previousMask[dim[0] - 1 - j, i] == 1):
                    self.mapAnnotation[frameID, i, 1] = dim[0] - 1 - j
                    # previousMask[dim[0] - 1 - j, i] = 100
                    condition = False

                elif j == limit:
                    self.mapAnnotation[frameID, i, 1] = self.mapAnnotation[frameID, i-1, 1]
                    condition = False
                    # previousMask[j, i] = 100

                j += 1

            j -= neighours + 1
            limit = j + 2*neighours
        j = 300
        for i in range(seed[1], xStart-1, -1):
            condition = True

            while condition == True:

                if (j < dim[0] and previousMask[dim[0] - 1 - j, i] == 1):
                    self.mapAnnotation[frameID, i, 1] = dim[0] - 1 - j
                    # previousMask[dim[0] - 1 - j, i] = 100
                    condition = False

                elif j == limit:
                    self.mapAnnotation[frameID, i, 1] = self.mapAnnotation[frameID, i-1, 1]
                    condition = False
                    # previousMask[j, i] = 100

                j += 1

            j -= neighours + 1
            limit = j + 2*neighours

        # For debug only

        # mask = np.empty((self.sequenceDimension[1], self.sequenceDimension[2], 3))
        # for k in range(xStart, xEnd):
        #     mask[round(self.mapAnnotation[0, k, 0]):round(self.mapAnnotation[0, k, 1]), k, 0] = 255
        #     mask[round(self.mapAnnotation[1, k, 0]):round(self.mapAnnotation[1, k, 1]), k, 2] = 255
        #
        # mask[:, :, 1] = previousMask


    def updateDynamicProg(self,
                          previousMask,
                          frameID,
                          scale):
        # print(2)
        imCost = previousMask.copy()
        imCost = np.square(imCost - 0.5) / 0.25

        # Image.fromarray(imCost).save("results/costMap.tiff")
        # plt.imsave("results/costMap.png", imCost, cmap='seismic')

        # 1/ --- we first treshold the image
        tresholdedImage = previousMask.copy()
        tresholdedImage[tresholdedImage > 0.01] = 1
        tresholdedImage[tresholdedImage < 1] = 0

        # 2/ --- we consider the biggest connexe region
        tresholdedImage = getBiggestConnexeRegion(tresholdedImage)

        # 3/ --- we compute the center of mass
        white_pixels = np.array(np.where(tresholdedImage == 1))

        # 4/ --- we compute the box which contain the connexe region to reduce the computational time and we extract the ROI
        xMin, xMax = np.min(white_pixels[1, :]), np.max(white_pixels[1, :])
        yMin, yMax = np.min(white_pixels[0, :]), np.max(white_pixels[0, :])
        start_point = (xMin, yMin)
        end_point = (xMax, yMax)
        thickness = 1
        # image = cv2.rectangle(im, start_point, end_point, thickness)
        # plt.imsave("results/ROI.png", image, cmap='gray')
        ROI = imCost[yMin:yMax, xMin:xMax]
        # Image.fromarray(ROI).save("results/ROICostMap.tiff"
        print("scale coefficient: ", scale)
        cumulativeCostMap, backTrackingMap = frontPropagation(costMap=ROI, scale=scale)
        LI, MA = tracking(cumulativeCostMap=cumulativeCostMap, backTrackingMap=backTrackingMap)

        LI+=yMin
        MA+=yMin

        self.mapAnnotation[frameID, self.borders['leftBorder']:self.borders['rightBorder']-1, 1] = LI
        self.mapAnnotation[frameID, self.borders['leftBorder']:self.borders['rightBorder']-1, 0] = MA

        # print(2)

    def IMT(self):

        xLeft = self.bordersROI['leftBorder']
        xRight = self.bordersROI['rightBorder']

        IMT = self.mapAnnotation[:, xLeft:xRight, 1] - self.mapAnnotation[:, xLeft:xRight, 0]
        # np.mean(IMT, axis=1)
        # # tmp = np.median(tmp, axis=1)
        return np.mean(IMT, axis=1), np.median(IMT, axis=1)

    def FWAutoInitialization(self,img, seed):

        # --- window of +/- neighbours pixels where the algorithm search the borders
        neighours = 10

        # --- the algorithm starts from the left to the right
        xStart = self.borders['leftBorder']
        xEnd = self.borders['rightBorder']

        # --- dimension of the mask
        dim = img.shape

        # --- random value to j but not to high. It the first y coordinate at the xStart position
        j = 5

        # --- j cannot exceed limit
        limit = dim[0]-1
        # --- delimitation of the LI boundary
        for i in range(seed[1]+1, xEnd+1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and img[j, i] == 1):
                    self.mapAnnotation[0, i, 0] = j
                    self.mapAnnotation[0, i, 1] = j
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    self.mapAnnotation[0, i, 0] = self.mapAnnotation[0, i-1, 0]
                    self.mapAnnotation[0, i, 1] = self.mapAnnotation[0, i - 1, 1]
                    condition = False

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2*neighours

        j = 5
        for i in range(seed[1], xStart-1, -1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and img[j, i] == 1):
                    self.mapAnnotation[0, i, 0] = j
                    self.mapAnnotation[0, i, 1] = j
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    self.mapAnnotation[0, i, 0] = self.mapAnnotation[0, i - 1, 0]
                    self.mapAnnotation[0, i, 1] = self.mapAnnotation[0, i - 1, 1]
                    condition = False

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighours + 1
            limit = j + 2 * neighours



def LoadBorders(borders_path):

    mat_b = scipy.io.loadmat(borders_path)
    right_b = mat_b['border_right']
    right_b = right_b[0, 0]-1
    left_b = mat_b['border_left']
    left_b = left_b[0, 0]-1

    return {"leftBorder": left_b,
            "rightBorder": right_b}

def frontPropagation(costMap, scale=10):

    '''
    :param costMap:
    :param seed:
    :param scale:
    :return:
    '''

    cumulativeCost = np.zeros(costMap.shape)
    cumulativeCost[:, 0] = costMap[:, 0]
    backProjMap = np.zeros(costMap.shape)
    # ---  we start from the left to the right
    deltaX = 1

    for x in range(costMap.shape[1]-1):

        for y in range(costMap.shape[0]):
            tmpCostMin = float('inf')
            for deltaY in range(round(scale), -round(scale)-1, -1):
                # print(deltaY)
                if y + deltaY < costMap.shape[0]-1 and y+deltaY >= 0:
                    tmpCost = cumulativeCost[y + deltaY, x] + (costMap[y, x + deltaX] + costMap[y + deltaY, x]) * (math.sqrt(1 + (deltaY / scale) ** 2))

                    if tmpCost < tmpCostMin:
                        tmpCostMin = tmpCost
                        cumulativeCost[y, x+deltaX] = tmpCostMin
                        backProjMap[y, x+deltaX]  = deltaY

    # Image.fromarray(backProjMap).save("results/backProjMap.tiff")
    # Image.fromarray(cumulativeCost).save("results/cumulativeCost.tiff")

    return cumulativeCost, backProjMap

def minLIMA(map):
    col = map[:, -30]
    min1 = np.argmin(col)
    # --- we modify the pixels' value around this minimum
    if min1-80>=0 and min1+80<col.shape[0]:
        col[min1-80:min1+80] = col.max()
    elif min1-80<0:
        col[0:min1 + 80] = col.max()
    else:
        col[min1-80:-1] = col.max()

    min2 = np.argmin(col)

    return min1, min2

def trackSeed(backTrackingMap, seed, map):
    boundary = np.zeros(backTrackingMap.shape[1])
    seedl=seed
    for k in range(backTrackingMap.shape[1]-30, backTrackingMap.shape[1]):
        boundary[k] = int(seed)
        map[seed, k] = 1
        seed+= int(backTrackingMap[seed, k])

    for k in range(backTrackingMap.shape[1]-30, -1, -1):
        boundary[k] = int(seedl)
        map[seedl, k] = 1
        seedl+= int(backTrackingMap[seedl, k])

    return map, boundary

# def trackSeed(backTrackingMap, seed, map):
#     boundary = np.zeros(backTrackingMap.shape[1])
#
#     for k in range(backTrackingMap.shape[1]-1, -1, -1):
#         boundary[k]=int(seed)
#         map[seed, k] = 1
#         seed += int(backTrackingMap[seed, k])
#
#
#
#     return map, boundary

def tracking(cumulativeCostMap, backTrackingMap):

    map = np.zeros(cumulativeCostMap.shape)
    seed1, seed2 = minLIMA(cumulativeCostMap)

    map, MA = trackSeed(backTrackingMap, seed1, map)
    map, LI = trackSeed(backTrackingMap, seed2, map)


    return LI, MA
    # plt.imsave("results/results.png", map, cmap='gray')