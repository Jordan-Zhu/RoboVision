import numpy as np


# Written 11/1/2016
# Label Line features (LabelLineFeature_v4)
# Pre-condition: Gets the depth image and list of line segments and parameters
# Post-condition: Returns a list of lines labeled as curvature edges or discontinuities
# ----------------
# We define curvature edges as lines (or curves) on the object and discontinuities as those lines
# on the outside edges of the object where you can see it touching the background.

global window_size

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
    else:
        # Horizontal line
        pt1, pt2, pt3, pt4, line = horizontal_line(line)
    window = [pt1, pt2, pt3, pt4]
    # Positive or negative side of the window is used
    # when we are checking which side the line is on the discontinuity
    return window, startpt, endpt, line


def create_windows(startpt, endpt, window):
    pt1, pt2, pt3, pt4 = window
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


def mask_length(line, window):
    range_start = 0
    range_end = 0
    if line[11] == 1:
        # Y range start / end
        range_start = min(int(window[0][0]), int(window[1][0]))
        range_end = max(int(window[0][0]), int(window[1][0]))
    elif line[11] == 2:
        # X range start / end
        range_start = min(int(window[0][1]), int(window[1][1]))
        range_end = max(int(window[0][1]), int(window[1][1]))
    return abs(range_start - range_end) * 2 * window_size


def roipoly(src, line, poly):
    mask = []
    dyy = line[0] - line[2]
    dxx = line[1] - line[3]

    if line[11] == 1:
        xfp = min(int(poly[0][1]), int(poly[1][1])) if dxx * dyy > 0 else max(int(poly[0][1]), int(poly[1][1]))
        mask_len = int(poly[3][1] - poly[0][1])
        y_range_start = min(int(poly[0][0]), int(poly[1][0]))
        y_range_end = max(int(poly[0][0]), int(poly[1][0]))

        for i in range(y_range_start, y_range_end):
            x0 = int(round(xfp))
            mask += list(src[i, x0:x0 + mask_len])
            step = (poly[1][1] - poly[0][1] + 0.0) / (poly[1][0] - poly[0][0] + 0.0)
            xfp += step
    elif line[11] == 2:
        yfp = min(int(poly[0][0]), int(poly[1][0])) if dxx * dyy > 0 else max(int(poly[0][0]), int(poly[1][0]))
        mask_len = int(poly[3][0] - poly[0][0])
        x_range_start = min(int(poly[0][1]), int(poly[1][1]))
        x_range_end = max(int(poly[0][1]), int(poly[1][1]))

        for i in range(x_range_start, x_range_end):
            y0 = int(round(yfp))
            mask += list(src[y0:y0 + mask_len, i])
            step = (poly[1][0] - poly[0][0] + 0.0) / (poly[1][1] - poly[0][1] + 0.0)
            yfp += step
    return mask
    # return src[poly[0][1]:poly[1][1], poly[0][0]:poly[1][0]]


def mean(arr):
    return sum(arr) / len(arr)


def obj_relation(depthimg, line, win_p, win_n):
    mask_p = roipoly(depthimg, line, win_p)
    mask_n = roipoly(depthimg, line, win_n)
    return 9 if mean(mask_p) > mean(mask_n) else 10


def label_line_features(depthimg, edgeimg, seglist, parameters):
    # Constants
    minlen = int(parameters["Cons_Lmin"])
    dis_thresh = int(parameters["thresh_label_dis"])
    global window_size
    window_size = int(parameters["label_win_sized"])

    # Get the lines which are longer than the minimum length
    desired_lines = [line for line in seglist if line[4] > minlen]
    # print(len(seglist))
    out = []

    for line in desired_lines:
        window, startpt, endpt, line = get_orientation(line)
        window, win_p, win_n = create_windows(startpt, endpt, window)
        roi = roipoly(edgeimg, line, window)

        len_mask = mask_length(line, window)
        dis_var = len(roi)/float(len_mask)
        if dis_var > dis_thresh:
            line[10] = obj_relation(depthimg, line, win_p, win_n)
        else:
            line[10] = 13
        line = np.reshape(line, (12, 1))
        # print(line.shape)
        out.append(line)

    return out
