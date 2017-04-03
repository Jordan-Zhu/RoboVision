import numpy as np
from utility import *


# Written 3/30/2017
# Label Line features (LabelLineFeature_v4)
# Pre-condition: Gets the depth image and list of line segments and parameters
# Post-condition: Returns a list of lines labeled as curvature edges or discontinuities
# ----------------
# We define curvature edges as lines (or curves) on the object and discontinuities as those lines
# on the outside edges of the object where you can see it touching the background.


global window_size


def vertical_line(line):
    line[11] = 1
    # [y ; x-ts]
    return [line[0], line[1] - window_size], \
           [line[0], line[1] + window_size], \
           [line[2], line[3] - window_size], \
           [line[2], line[3] + window_size]


def horizontal_line(line):
    line[11] = 2
    # [y-ts ; x]
    return [line[0] - window_size, line[1]], \
           [line[0] + window_size, line[1]], \
           [line[2] - window_size, line[3]], \
           [line[2] + window_size, line[3]]


def get_orientation(line):
    startpt = [line[0], line[1]]
    endpt = [line[2], line[3]]
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])

    if dy > dx or dy == dx:
        pt1, pt2, pt3, pt4 = vertical_line(line)
    else:
        pt1, pt2, pt3, pt4 = horizontal_line(line)
    return pt1, pt2, pt3, pt4, startpt, endpt


def create_windows(pt1, pt2, pt3, pt4, startpt, endpt):
    temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
    temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
    if temp1 > temp2:
        window = [pt1, pt3, pt4, pt2]
        win_p = [startpt, endpt, pt4, pt2]
        win_n = [pt1, pt3, endpt, startpt]
    else:
        window = [pt1, pt4, pt3, pt2]
        win_p = [startpt, pt4, endpt, pt2]
        win_n = [pt1, endpt, pt3, startpt]
    return window, win_p, win_n


def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)
    win = swap_indices(poly).astype(int)

    cv2.fillConvexPoly(mask, win, 255)  # Create the ROI
    return mask


def label_line_features(depth_img, edge_img, seg_list, parameters):
    minlen = int(parameters["Cons_Lmin"])
    dis_thresh = int(parameters["thresh_label_dis"])
    global window_size
    window_size = int(parameters["label_win_sized"])

    zeros = np.zeros((seg_list.shape[0], 2))
    res = np.hstack((seg_list, zeros))
    count = 0
    for i, line in enumerate(res):
        if line[4] > minlen:
            count += 1
            pt1, pt2, pt3, pt4, startpt, endpt = get_orientation(line)
            win, win_p, win_n = create_windows(pt1, pt2, pt3, pt4, startpt, endpt)
            mask = roipoly(edge_img, win)
            edgels = edge_img * mask
            disc_var = cv2.countNonZero(edgels) / cv2.countNonZero(mask)
            if disc_var > dis_thresh:
                mask_pos = roipoly(edge_img, win_p)
                mask_neg = roipoly(edge_img, win_n)
                edge_dis_p = depth_img * mask_pos
                edge_dis_n = depth_img * mask_neg
                if cv2.countNonZero(edge_dis_p) == 0:
                    mean_p = 0
                else:
                    mean_p = sum(depth_img[np.nonzero(edge_dis_p)]) / cv2.countNonZero(edge_dis_p)
                if cv2.countNonZero(edge_dis_n) == 0:
                    mean_n = 0
                else:
                    mean_n = sum(depth_img[np.nonzero(edge_dis_n)]) / cv2.countNonZero(edge_dis_n)

                if mean_p > mean_n:
                    line[10] = 9
                else:
                    line[10] = 10
            else:
                line[10] = 13   # Line is a curvature
    print('count:', count)

    return res
