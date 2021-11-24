import numpy as np
import cv2

def patch_extraction(img,
                     manualDel,
                     borders,
                     widthWindow,
                     overlay,
                     nameSeq,
                     resizeCol,
                     skipped_sequences,
                     spatial_res_y,
                     spatial_res_x,
                     desiredSpatialResolution,
                     imgNumber):

    '''
    This function extracts patches (mask and image)

    Parameters:
        img: first frame of the sequence
        manualDel: manual delineation
        borders: windows where the image is exploitable
        widthWindow: width of the sliding windows (in other work the width of the patch)
        overlay: number of pixel that you want to superimpose
        nameSeq: name of the sequence
        resizeCol: resize col by a factor 10
        dic_storage: dictionnary where data are stored
        skipped_sequences: text file to know which sequences are skipped in function of widthWindow
        spatial_res_y: spatial resolution of the sequence
        spatialResolution: desired spatial resolution

    Returns:
        dic_datas (dic): dictionnary containing patches for one patient only
        imgNumber (dic): total of computed images
    '''

    dim = img.shape
    leftB = borders[0]  # this is not a constant\fancyfoot[C]{}
    rightB = borders[1]  # this is a constant

    height_i = dim[0]
    width_i = dim[1]

    condition = True
    inc = 0

    LI = manualDel[0]
    MA = manualDel[1]


    if resizeCol == True:

        height_patch = 512

        scale = (spatial_res_y * 10000) / desiredSpatialResolution
        height_tmp = round(scale * height_i)

        scale = height_tmp / height_i
        height_i = height_tmp

        LI = scale * LI
        MA = scale * MA

        img = cv2.resize(img.astype(np.float32), (width_i, height_i), interpolation=cv2.INTER_LINEAR)
        # img = cv2.resize(img, (width_i, height_i), interpolation=cv2.INTER_LINEAR)
    else:
        height_patch = 128
        scale = 1

    tmpStop = True
    if (rightB - leftB) >= widthWindow:

        dic_datas = {'patch_mask': {},
                     'patch_Image_org': {},
                     'spatial_resolution': {}}

        while condition:
            # -- declaration of the patches
            patch_m = np.zeros((height_i, widthWindow))

            LItmp = LI[leftB:leftB + widthWindow, 0]
            MAtmp = MA[leftB:leftB + widthWindow, 0]

            patch_img = img[:, leftB:leftB + widthWindow]

            for i in range(widthWindow):
                LIVal = round(LItmp[i])
                MAVal = round(MAtmp[i])
                patch_m[LIVal:MAVal, i] = 255
                patch_img[LIVal, i] = 255
                patch_img[MAVal, i] = 255

            mean = round(np.mean(np.concatenate((LItmp, MAtmp))))

            lim_inf = int(mean - height_patch / 2)
            lim_sup = int(mean + height_patch / 2)

            patch_m_center = patch_m[lim_inf:lim_sup, :]
            patch_m_bottom = patch_m[lim_inf - int(10 * scale):lim_sup - int(10 * scale), :]
            patch_m_top = patch_m[lim_inf + int(10 * scale):lim_sup + int(10 * scale), :]

            patch_img_center = patch_img[lim_inf:lim_sup, :]
            tmpMin = np.min(patch_img_center)
            patch_img_center = (patch_img_center - tmpMin)
            tmpMax = np.max(patch_img_center)
            coef = 255 / tmpMax
            patch_img_center = patch_img_center * coef

            patch_img_bottom = patch_img[lim_inf - int(10 * scale):lim_sup - int(10 * scale), :]
            tmpMin = np.min(patch_img_bottom)
            patch_img_bottom = (patch_img_bottom - tmpMin)
            tmpMax = np.max(patch_img_bottom)
            coef = 255 / tmpMax
            patch_img_bottom = patch_img_bottom * coef

            patch_img_top = patch_img[lim_inf + int(10 * scale):lim_sup + int(10 * scale), :]
            tmpMin = np.min(patch_img_top)
            patch_img_top = (patch_img_top - tmpMin)
            tmpMax = np.max(patch_img_top)
            coef = 255 / tmpMax
            patch_img_top = patch_img_top * coef

            dic_datas['patch_mask']['patch_center_' + str(inc) + "_" + nameSeq] = patch_m_center.astype(np.uint8)
            dic_datas['patch_mask']['patch_bottom_' + str(inc) + "_" + nameSeq] = patch_m_bottom.astype(np.uint8)
            dic_datas['patch_mask']['patch_top_' + str(inc) + "_" + nameSeq] = patch_m_top.astype(np.uint8)

            dic_datas['patch_Image_org']['patch_center_' + str(inc) + "_" + nameSeq] = patch_img_center.astype(np.float32)
            dic_datas['patch_Image_org']['patch_bottom_' + str(inc) + "_" + nameSeq] = patch_img_bottom.astype(np.float32)
            dic_datas['patch_Image_org']['patch_top_' + str(inc) + "_" + nameSeq] = patch_img_top.astype(np.float32)

            dic_datas['spatial_resolution']['patch_center_' + str(inc) + "_" + nameSeq] = np.asarray([spatial_res_x, spatial_res_y / scale])  # (deltaX, deltaY)
            dic_datas['spatial_resolution']['patch_bottom_' + str(inc) + "_" + nameSeq] = np.asarray([spatial_res_x, spatial_res_y / scale])
            dic_datas['spatial_resolution']['patch_top_' + str(inc) + "_" + nameSeq] = np.asarray([spatial_res_x, spatial_res_y / scale])

            inc = inc + 1
            imgNumber = imgNumber + 3

            if tmpStop == False:
                condition = False
            elif ((leftB + overlay + widthWindow) < rightB):
                leftB = leftB + overlay
            elif (leftB + overlay + widthWindow > rightB):
                leftB = rightB - widthWindow
                tmpStop = False
            else:
                condition = False

        return dic_datas, imgNumber
    else:

        tmp = rightB - leftB
        skipped_sequences.write("Patient name: " + nameSeq + ", width window: " + str(tmp) + "\n")
        return "skipped", imgNumber