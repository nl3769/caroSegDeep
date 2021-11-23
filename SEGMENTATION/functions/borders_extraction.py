import numpy as np
import math
import  matplotlib.pyplot as plt
# from functions.backPropagation import computeGradientDescent1D
from PIL import Image

def frontPropagation(costMap, scale=15):

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
            for deltaY in range(scale, -scale-1, -1):

                if y + deltaY < costMap.shape[0]-1 and y+deltaY >= 0:
                    tmpCost = cumulativeCost[y + deltaY, x] + (costMap[y, x + deltaX] + costMap[y + deltaY, x]) * (math.sqrt(1 + (deltaY / scale) ** 2))
                    # tmpCost = cumulativeCost[y + deltaY, x] + (costMap[y, x + deltaX]) * (math.sqrt(1 + (deltaY / scale) ** 2))
                    if tmpCost < tmpCostMin:
                        tmpCostMin = tmpCost
                        cumulativeCost[y, x+deltaX] = tmpCostMin
                        backProjMap[y, x+deltaX]  = deltaY


    # # --- test
    # deltaX = 1
    # midlePos = round(costMap.shape[1]/2)
    # for x in range(midlePos, costMap.shape[1]-1):
    #
    #     for y in range(costMap.shape[0]):
    #         tmpCostMin = float('inf')
    #         for deltaY in range(scale, -scale-1, -1):
    #             # print(deltaY)
    #             if y + deltaY < costMap.shape[0]-1 and y+deltaY >= 0:
    #                 tmpCost = cumulativeCost[y + deltaY, x] + (costMap[y, x + deltaX] + costMap[y + deltaY, x]) * (math.sqrt(1 + (deltaY / scale) ** 2))
    #
    #                 if tmpCost < tmpCostMin:
    #                     tmpCostMin = tmpCost
    #                     cumulativeCost[y, x+deltaX] = tmpCostMin
    #                     backProjMap[y, x+deltaX]  = deltaY
    # deltaX = -1
    #
    # for x in range(midlePos, -1, -1):
    #
    #     for y in range(costMap.shape[0]):
    #         tmpCostMin = float('inf')
    #         for deltaY in range(scale, -scale-1, -1):
    #             # print(deltaY)
    #             if y + deltaY < costMap.shape[0]-1 and y+deltaY >= 0:
    #                 tmpCost = cumulativeCost[y + deltaY, x] + (costMap[y, x + deltaX] + costMap[y + deltaY, x]) * (math.sqrt(1 + (deltaY / scale) ** 2))
    #
    #                 if tmpCost < tmpCostMin:
    #                     tmpCostMin = tmpCost
    #                     cumulativeCost[y, x+deltaX] = tmpCostMin
    #                     backProjMap[y, x+deltaX]  = deltaY

    # Image.fromarray(backProjMap).save("backProjMap.tiff")
    # Image.fromarray(cumulativeCost).save("cumulativeCost.tiff")

    return cumulativeCost, backProjMap

def tracking(cumulativeCostMap, backTrackingMap):

    map = np.zeros(cumulativeCostMap.shape)
    LI = np.zeros(cumulativeCostMap.shape[1])
    MA = np.zeros(cumulativeCostMap.shape[1])
    seed1, seed2 = minLIMA(cumulativeCostMap)
    # seed1, seed2 = computeGradientDescent1D(lastColDer, lastCol, LI = True), computeGradientDescent1D(lastColDer, lastCol, MA = True)
    # LI = computeGradientDescent1D(lastColDer, lastCol, LI = True)
    # MA = computeGradientDescent1D(lastColDer, lastCol, MA = True)
    map, LI = trackSeed(backTrackingMap, seed1, map, LI)
    map, MA = trackSeed(backTrackingMap, seed2, map, MA)

    return LI, MA
    # plt.imsave("results.png", map, cmap='gray')

def trackSeed(backTrackingMap, seed, map, arr):

    seed_cp = seed
    for k in range(backTrackingMap.shape[1]-30, backTrackingMap.shape[1]):
        map[seed, k] = 1
        arr[k] = seed
        seed+= int(backTrackingMap[seed, k])

    for k in range(backTrackingMap.shape[1]-30, -1, -1):
        map[seed_cp, k] = 1
        arr[k] = seed_cp
        seed_cp+= int(backTrackingMap[seed_cp, k])

    return map, arr

def minLIMA(map):
    prop = -30
    col = map[:, prop].copy()
    min1 = np.argmin(col)
    # --- we modify the pixels' value around this minimum
    changeValue = 80
    if min1-changeValue>=0 and min1+changeValue<col.shape[0]:
        col[min1-changeValue:min1+changeValue] = col.max()
    elif min1-changeValue<0:
        col[0:min1 + changeValue] = col.max()
    else:
        col[min1-changeValue:-1] = col.max()

    min2 = np.argmin(col)

    return min1, min2