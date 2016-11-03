import numpy as np


# Written 11/1/2016
# Label Line features
# Pre-condition: Gets the depth image and list of line segments and parameters
# Post-condition: Returns a list of lines labeled as curvature edges or discontinuities
# ----------------
# We define curvature edges as lines (or curves) on the object and discontinuities as those lines
# on the outside edges of the object where you can see it touching the background.


def vertical_line(line):
    line = np.append(line, [0, 1])
    return [line[0], line[1] - window_size], \
           [line[0], line[1] + window_size], \
           [line[2], line[3] - window_size], \
           [line[2], line[3] + window_size], line


def horizontal_line(line):
    line = np.append(line, [0, 2])  # fill in the spot for +/- side of the line and horizontal line in index 11
    return [line[0] - window_size, line[1]], \
           [line[0] + window_size, line[1]], \
           [line[2] - window_size, line[3]], \
           [line[2] + window_size, line[3]], line


def get_orientation(line):
    startpt = [line[0], line[1]]
    endpt = [line[2], line[3]]
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])

    # determine if the line is vertical or horizontal
    # and set up the coords for the region of interest
    if dy > dx or dy == dx:
        # Vertical line
        pt1, pt2, pt3, pt4, line = vertical_line(line)
        # pt1 = [line[0], line[1] - mask_size]
        # pt2 = [line[0], line[1] + mask_size]
        # pt3 = [line[2], line[3] - mask_size]
        # pt4 = [line[2], line[3] + mask_size]
        # line.append(0)  # fill in the spot for +/- side of the line
        # line.append(1)	# tag for vertical lines in index 11
        win_pos = [startpt, endpt, pt4, pt2]
        win_neg = [pt1, pt3, endpt, startpt]
    else:
        # Horizontal line
        pt1, pt2, pt3, pt4, line = horizontal_line(line)
        # pt1 = [line[0] - mask_size, line[1]]
        # pt2 = [line[0] + mask_size, line[1]]
        # pt3 = [line[2] - mask_size, line[3]]
        # pt4 = [line[2] + mask_size, line[3]]
        # line.append(0)  # fill in the spot for +/- side of the line
        # line.append(2)  # horizontal line in index 11
        win_pos = [startpt, pt4, endpt, pt2]
        win_neg = [pt1, endpt, pt3, startpt]
    window = [pt1, pt2, pt3, pt4]
    # Positive or negative side of the window is used
    # when we are checking which side the line is on the discontinuity
    return window, win_pos, win_neg, line


def roipoly(src, poly):
    return src[poly[0][1]:poly[2][1], poly[0][0]:poly[1][0]]


def mean(arr):
    # print(arr.type)
    return np.sum(arr) / arr.shape[0]


def obj_relation(depthimg, win_p, win_n):
    mask_p = roipoly(depthimg, win_p)
    mask_n = roipoly(depthimg, win_n)
    # print("mean p ", mean(mask_p))
    return 9 if mean(mask_p) > mean(mask_n) else 10


def label_line_features(depthimg, edgeimg, seglist, parameters):
    # Constants
    minlen = int(parameters["Cons_Lmin"])
    dis_thresh = int(parameters["thresh_label_dis"])
    global window_size
    window_size = int(parameters["label_win_sized"])

    # Get the lines which are longer than the minimum length
    desired_lines = [line for line in seglist if line[4] > minlen]
    out = []

    for line in desired_lines:
        window, win_p, win_n, line = get_orientation(line)
        roi = roipoly(edgeimg, window)

        # temp1 = np.linalg.norm(np.subtract((np.add(ptd1, ptd3) / 2.0), (np.add(ptd2, ptd4) / 2.0)))
        # temp2 = np.linalg.norm(np.subtract((np.add(ptd1, ptd4) / 2.0), (np.add(ptd2, ptd3) / 2.0)))
        # if (norm(((ptd1 + ptd3) / 2) - ((ptd2 + ptd4) / 2)) > norm(((ptd1 + ptd4) / 2) - ((ptd2 + ptd3) / 2)))

        p1 = [line[0], line[1]]
        p2 = [line[2], line[3]]
        ptd1, ptd2, ptd3, ptd4 = window
        temp1 = np.linalg.norm(np.subtract((np.add(ptd1, ptd3) / 2.0), (np.add(ptd2, ptd4) / 2.0)))
        temp2 = np.linalg.norm(np.subtract((np.add(ptd1, ptd4) / 2.0), (np.add(ptd2, ptd3) / 2.0)))
        if temp1 > temp2:
            vxd = [ptd1, ptd3, ptd4, ptd2]
            winp = [p1, p2, ptd4, ptd2]
            winn = [ptd1, ptd3, p2, p1]
        else:
            vxd = [ptd1, ptd4, ptd3, ptd2]
            winp = [p1, ptd4, p2, ptd2]
            winn = [ptd1, p2, ptd3, p1]
        len_mask = abs(ptd1[0] - ptd1[1]) * 2 * window_size if line[11] == 1 else abs(vxd[0][1] - vxd[1][1]) * 2 * window_size
        # len_mask = np.linalg.norm(window[0][0] - window[1][0]) * 2 * window_size if line[11] == 1 else np.linalg.norm(window[0][1] - window[1][1]) * 2 * window_size
        dis_var = len(roi)/float(len_mask)
        if dis_var > dis_thresh:
            line[10] = obj_relation(depthimg, win_p, win_n)
        else:
            line[10] = 13
        line = np.reshape(line, (12, 1))
        # print(line.shape)
        out.append(line)

    return out
