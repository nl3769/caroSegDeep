"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import numpy as np
import cv2
import _io

def patch_preprocessing(patch: np.ndarray):
    """ Spread grayscale values in the interval [0, 255]. """
    min_ = np.min(patch)
    patch=(patch-min_)
    max_ = np.max(patch)
    coef_ =255/max_
    patch_img_center=patch * coef_

    return patch_img_center

def patch_extraction_wall(img: np.ndarray,
                          manual_del: tuple,
                          borders: tuple,
                          width_window: int,
                          overlay: _io.TextIOWrapper,
                          name_seq: str,
                          resize_col: bool,
                          skipped_sequences,
                          spatial_res_y: float,
                          spatial_res_x: float,
                          desired_spatial_res: int,
                          img_nb: int):

    """ Extracts patches (mask and image), and write skipped images in .txt. An image is skipped if its annotation is done on less than width_window. """
    # --- get information
    dim = img.shape
    left_border = borders[0]
    right_border = borders[1]
    height_i = dim[0]
    width_i = dim[1]
    LI = manual_del[0]
    MA = manual_del[1]
    # --- initialization
    condition = True
    inc = 0

    if resize_col == True:
        # --- vertical interpolation to achieve a uniform vertical pixel size
        height_patch = 512
        scale = (spatial_res_y * 10000) / desired_spatial_res
        height_tmp = round(scale * height_i)
        scale = height_tmp / height_i
        height_i = height_tmp
        LI = scale * LI
        MA = scale * MA
        img = cv2.resize(img.astype(np.float32), (width_i, height_i), interpolation=cv2.INTER_LINEAR)
    else:
        height_patch = 128
        scale = 1

    stop_ = True # variable used to stop the extraction
    if (right_border - left_border) >= width_window:
        # --- dictionary that contains the data
        dic_datas = {'patch_mask': {},
                     'patch_Image_org': {},
                     'spatial_resolution': {}}
        while condition:
            # -- data extraction
            patch_m = np.zeros((height_i, width_window))
            LI_ = LI[left_border:left_border + width_window, 0]
            MA_ = MA[left_border:left_border + width_window, 0]
            patch_img = img[:, left_border:left_border + width_window]
            # -- mask generation
            for i in range(width_window):
                LI_val = round(LI_[i])
                MA_val = round(MA_[i])
                patch_m[LI_val:MA_val, i] = 255
                # --- uncomment below to see the position of the LI/MA interface
                # patch_img[LI_val, i] = 255
                # patch_img[MA_val, i] = 255
            # --- compute average position in order to center the patch
            mean = round(np.mean(np.concatenate((LI_, MA_))))
            # --- compute the position of the patch/mask
            lim_inf = int(mean - height_patch / 2)
            lim_sup = int(mean + height_patch / 2)
            # --- get masks
            patch_m_center = patch_m[lim_inf:lim_sup, :]
            patch_m_bottom = patch_m[lim_inf - int(10 * scale):lim_sup - int(10 * scale), :]
            patch_m_top = patch_m[lim_inf + int(10 * scale):lim_sup + int(10 * scale), :]
            # --- get patch an apply preprocessing
            patch_img_center = patch_img[lim_inf:lim_sup, :]
            patch_img_center=patch_preprocessing(patch_img_center)
            # --- get patch an apply preprocessing
            patch_img_bottom=patch_img[lim_inf - int(10 * scale):lim_sup - int(10 * scale), :]
            patch_img_bottom=patch_preprocessing(patch_img_bottom)
            # --- get patch an apply preprocessing
            patch_img_top = patch_img[lim_inf + int(10 * scale):lim_sup + int(10 * scale), :]
            patch_img_top=patch_preprocessing(patch_img_top)
            # --- add to dic_datas
            # mask
            dic_datas['patch_mask']['patch_center_' + str(inc) + "_" + name_seq] = patch_m_center.astype(np.uint8)
            dic_datas['patch_mask']['patch_bottom_' + str(inc) + "_" + name_seq] = patch_m_bottom.astype(np.uint8)
            dic_datas['patch_mask']['patch_top_' + str(inc) + "_" + name_seq] = patch_m_top.astype(np.uint8)
            # image
            dic_datas['patch_Image_org']['patch_center_' + str(inc) + "_" + name_seq] = patch_img_center.astype(np.float32)
            dic_datas['patch_Image_org']['patch_bottom_' + str(inc) + "_" + name_seq] = patch_img_bottom.astype(np.float32)
            dic_datas['patch_Image_org']['patch_top_' + str(inc) + "_" + name_seq] = patch_img_top.astype(np.float32)
            # pixel size
            dic_datas['spatial_resolution']['patch_center_' + str(inc) + "_" + name_seq] = np.asarray([spatial_res_x, spatial_res_y/scale])  # (deltaX, deltaY)
            dic_datas['spatial_resolution']['patch_bottom_' + str(inc) + "_" + name_seq] = np.asarray([spatial_res_x, spatial_res_y/scale])
            dic_datas['spatial_resolution']['patch_top_' + str(inc) + "_" + name_seq] = np.asarray([spatial_res_x, spatial_res_y/scale])
            # --- update variables
            inc=inc+1
            img_nb=img_nb+3 # since three patches are extracted
            # --- check if the condition is always verified
            if stop_ == False:
                condition = False
            elif ((left_border + overlay + width_window) < right_border):
                left_border = left_border + overlay
            elif (left_border + overlay + width_window > right_border):
                left_border = right_border - width_window
                stop_ = False
            else:
                condition = False

        return dic_datas, img_nb
    else:
        # --- write a patient that cannot be used in the dataset (typically because the width of the annotation is too small).
        diff_ = right_border - left_border
        skipped_sequences.write("Patient name: " + name_seq + ", width window: " + str(diff_) + "\n")
        return "skipped", img_nb

def patch_extraction_far_wall(img,
                              manual_del,
                              borders,
                              width_window,
                              overlay,
                              name_seq,
                              skipped_sequences,
                              spatial_res_y,
                              spatial_res_x,
                              img_nb):

    """ Extract full-height and width_window width in the ROI defined by experts. """

    dim = img.shape
    leftB = borders[0]   # this is not a constant\fancyfoot[C]{}
    rightB = borders[1]  # this is a constant

    height_i = 512
    scale_coef = height_i/dim[0]
    width_i = dim[1]

    condition = True
    inc = 0

    LI = manual_del[0]
    MA = manual_del[1]

    img = cv2.resize(img.astype(np.float32), (width_i, height_i), interpolation=cv2.INTER_LINEAR)

    tmpStop = True
    if (rightB-leftB) >= width_window:

        dic_datas = {'patch_mask': {},
                     'patch_Image_org': {},
                     'spatial_resolution': {}}

        while condition:

            LI_ = LI[leftB:leftB + width_window]
            MA_ = MA[leftB:leftB + width_window]
            middlePos = scale_coef*(LI_ + MA_)/2

            patch_img = img[:, leftB:leftB + width_window]
            tmpMin = np.min(patch_img)
            patch_img = (patch_img - tmpMin)
            tmpMax = np.max(patch_img)
            coef = 255 / tmpMax
            patch_img = patch_img * coef

            patch_mask = np.zeros(patch_img.shape)

            for k in range(middlePos.shape[0]):
                patch_mask[round(middlePos[k, 0]):, k] = 255

            dic_datas['patch_mask']['patch_'+ str(inc) + "_" + name_seq] = patch_mask.astype(np.uint8)
            dic_datas['patch_Image_org']['patch_'+ str(inc) + "_" + name_seq] = patch_img.astype(np.uint8)
            dic_datas['spatial_resolution']['patch_'+ str(inc) + "_" + name_seq] = np.asarray([spatial_res_x, spatial_res_y/scale_coef])

            inc+=1
            img_nb+=1

            if tmpStop == False:
                condition = False
            elif ((leftB + overlay + width_window) < rightB):
                leftB = leftB + overlay
            elif  (leftB + overlay + width_window > rightB):
                leftB = rightB - width_window
                tmpStop = False
            else:
                condition = False

        return dic_datas, img_nb
    else:

        tmp = rightB-leftB
        skipped_sequences.write("Patient name: " + name_seq + ", width window: " + str(tmp) +  "\n")
        return "skipped", img_nb