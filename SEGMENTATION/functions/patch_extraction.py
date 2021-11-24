import numpy as np
import cv2
import _io

def patch_extraction(img: np.ndarray,
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

    ''' This function extracts patches (mask and image), and write skipped images in .txt. An image is skipped if its annoation is done on less than width_window pixels. '''


    dim = img.shape
    left_border = borders[0]
    right_border = borders[1]

    height_i = dim[0]
    width_i = dim[1]

    condition = True
    inc = 0

    LI = manual_del[0]
    MA = manual_del[1]


    if resize_col == True:

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

    stop_ = True
    if (right_border - left_border) >= width_window:

        dic_datas = {'patch_mask': {},
                     'patch_Image_org': {},
                     'spatial_resolution': {}}

        while condition:
            # -- declaration of the patches
            patch_m = np.zeros((height_i, width_window))

            LI_ = LI[left_border:left_border + width_window, 0]
            MA_ = MA[left_border:left_border + width_window, 0]

            patch_img = img[:, left_border:left_border + width_window]

            for i in range(width_window):
                LI_val = round(LI_[i])
                MA_val = round(MA_[i])
                patch_m[LI_val:MA_val, i] = 255
                patch_img[LI_val, i] = 255
                patch_img[MA_val, i] = 255

            mean = round(np.mean(np.concatenate((LI_, MA_))))

            lim_inf = int(mean - height_patch / 2)
            lim_sup = int(mean + height_patch / 2)

            # --- get masks
            patch_m_center = patch_m[lim_inf:lim_sup, :]
            patch_m_bottom = patch_m[lim_inf - int(10 * scale):lim_sup - int(10 * scale), :]
            patch_m_top = patch_m[lim_inf + int(10 * scale):lim_sup + int(10 * scale), :]

            # --- get patch an apply preprocessing
            patch_img_center = patch_img[lim_inf:lim_sup, :]
            min_ = np.min(patch_img_center)
            patch_img_center = (patch_img_center - min_)
            max_ = np.max(patch_img_center)
            coef_ = 255 / max_
            patch_img_center = patch_img_center * coef_
            # --- get patch an apply preprocessing
            patch_img_bottom = patch_img[lim_inf - int(10 * scale):lim_sup - int(10 * scale), :]
            min_ = np.min(patch_img_bottom)
            patch_img_bottom = (patch_img_bottom - min_)
            max_ = np.max(patch_img_bottom)
            coef_ = 255 / max_
            patch_img_bottom = patch_img_bottom * coef_
            # --- get patch an apply preprocessing
            patch_img_top = patch_img[lim_inf + int(10 * scale):lim_sup + int(10 * scale), :]
            min_ = np.min(patch_img_top)
            patch_img_top = (patch_img_top - min_)
            max_ = np.max(patch_img_top)
            coef_ = 255 / max_
            patch_img_top = patch_img_top * coef_

            # --- add to dic_datas
            dic_datas['patch_mask']['patch_center_' + str(inc) + "_" + name_seq] = patch_m_center.astype(np.uint8)
            dic_datas['patch_mask']['patch_bottom_' + str(inc) + "_" + name_seq] = patch_m_bottom.astype(np.uint8)
            dic_datas['patch_mask']['patch_top_' + str(inc) + "_" + name_seq] = patch_m_top.astype(np.uint8)

            dic_datas['patch_Image_org']['patch_center_' + str(inc) + "_" + name_seq] = patch_img_center.astype(np.float32)
            dic_datas['patch_Image_org']['patch_bottom_' + str(inc) + "_" + name_seq] = patch_img_bottom.astype(np.float32)
            dic_datas['patch_Image_org']['patch_top_' + str(inc) + "_" + name_seq] = patch_img_top.astype(np.float32)

            dic_datas['spatial_resolution']['patch_center_' + str(inc) + "_" + name_seq] = np.asarray([spatial_res_x, spatial_res_y/scale])  # (deltaX, deltaY)
            dic_datas['spatial_resolution']['patch_bottom_' + str(inc) + "_" + name_seq] = np.asarray([spatial_res_x, spatial_res_y/scale])
            dic_datas['spatial_resolution']['patch_top_' + str(inc) + "_" + name_seq] = np.asarray([spatial_res_x, spatial_res_y/scale])

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

        diff_ = right_border - left_border
        skipped_sequences.write("Patient name: " + name_seq + ", width window: " + str(diff_) + "\n")
        return "skipped", img_nb