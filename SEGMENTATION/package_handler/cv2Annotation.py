'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import cv2
import numpy as np

import matplotlib.pyplot as plt
from scipy import interpolate

class cv2Annotation:

    def __init__(self, windowName: str, image: np.ndarray):

        self.windowName = windowName
        self.backup = image.copy()
        self.img1 = image.copy()
        self.img = self.img1.copy()
        self.ROI = []
        self.channel = image[:, :, 0].copy()

        self.dim = image.shape

        print(self.dim)

        self.xLeft = 0
        self.xRight = 0
        self.xLeftSeg = 0
        self.xRighSeg = 0

        self.wallPosition = []
        self.curr_pt = []
        self.point = []
        self.restart = False

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(windowName, image)

        self.colorCirle = (0, 0, 255)  # BGR format

        self.thicknessCircle = -1
        self.radiusCircle = 1

        self.stop = False

    def select_points(self, event, x: int, y: int, flags: int, param):

        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON:
            self.stop = True

        elif event == cv2.EVENT_MBUTTONUP or self.restart == False:

            self.restart = True
            self.img = self.backup.copy()
            self.img1 = self.backup.copy()
            self.channel = self.backup[:, :, 0].copy()
            self.point = []
            self.curr_pt = []

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x, y])

            cv2.circle(self.img,
                       (x, y),
                       self.radiusCircle,
                       self.colorCirle,
                       self.thicknessCircle)

            if len(self.point) < 3:

                self.img[:, x, 0:2] = 0
                self.img[:, x, 2] = 255

                if len(self.point) == 1:
                    self.xLeft = self.point[0][0]
                if len(self.point) == 2:
                    self.xRight = self.point[1][0]
                    self.ROI = self.img.copy()

            elif len(self.point) < 5:

                self.img[:, x, 0:2] = 255
                self.img[:, x, 2] = 0

                if len(self.point) == 3:
                    self.xLeftSeg = self.point[2][0]
                    leftB = self.xLeftSeg + 128

                    if leftB < self.xRight:
                        leftB = self.xRight
                    if leftB >= self.dim[1]:
                        self.restart = False

                    self.img[:, self.xLeftSeg:leftB, 2] = 127

                if len(self.point) == 4:

                    self.xRightSeg = self.point[3][0]

                    self.ROI[:, self.xLeftSeg, 0:2] = 255
                    self.ROI[:, self.xRightSeg, 0:2] = 255
                    self.ROI[:, self.xLeftSeg, 2] = 0
                    self.ROI[:, self.xRightSeg, 2] = 0

                    self.img = self.ROI.copy()

                    xnew = np.arange(self.xLeftSeg, self.xRightSeg, 1)
                    xP = np.asarray(self.point)[:, 0]
                    yP = np.asarray(self.point)[:, 1]

                    # ---- it is necessary to sort the table to call interpolate.splrep
                    sort = xP.argsort()
                    xSorted = np.zeros(xP.shape[0])
                    ySorted = np.zeros(xP.shape[0])

                    for k in range(sort.shape[0]):
                        xSorted[k] = xP[sort[k]]
                        ySorted[k] = yP[sort[k]]

                    # --- cubic spline interpolation
                    tck = interpolate.splrep(xSorted, ySorted, s=0)
                    ynew = interpolate.splev(xnew, tck, der=0)

                    self.wallPosition = ynew.copy()

                    # --- we update the image
                    lastChannel = self.channel.copy()
                    for k in range(ynew.shape[0]):
                        lastChannel[round(ynew[k]), xnew[k]] = 255

                    self.img[:, :, 2] = lastChannel

            else:

                xnew = np.arange(self.xLeftSeg, self.xRightSeg, 1)
                xP = np.asarray(self.point)[:, 0]
                yP = np.asarray(self.point)[:, 1]

                sort = xP.argsort()
                xSorted = np.zeros(xP.shape[0])
                ySorted = np.zeros(xP.shape[0])

                for k in range(sort.shape[0]):
                    xSorted[k] = xP[sort[k]]
                    ySorted[k] = yP[sort[k]]

                # --- cubic spline interpolation
                tck = interpolate.splrep(xSorted, ySorted, s=0)
                ynew = interpolate.splev(xnew, tck, der=0)

                # --- we update the image
                lastChannel = self.channel.copy()
                for k in range(ynew.shape[0]):
                    lastChannel[round(ynew[k]), xnew[k]] = 255

                self.wallPosition = ynew.copy()
                self.img[:, :, 2] = lastChannel

    def getpt(self, img=None):

        '''
        :return self.wallPosition: the position of the far in an np.array with dimension (1, self.xRightSeg-self.xLeftSeg)
        :return self.img: the image one which we estimate the far wall
        :return [self.xLeft, self.xRight]: the region of interest
        :return [self.xLeftSeg, self.xRightSeg]: are of 128 at least to launch the segmentation
        '''

        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()

        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowName, self.img)
        cv2.setMouseCallback(self.windowName, self.select_points)
        self.point = []

        while (1):
            cv2.imshow(self.windowName, self.img)
            k = cv2.waitKey(2)

            if self.stop == True:
                break

        cv2.setMouseCallback(self.windowName, lambda *args: None)
        cv2.destroyWindow(self.windowName)

        return self.wallPosition, self.img, [self.xLeft, self.xRight], [self.xLeftSeg, self.xRightSeg]