import numpy as np
import cv2
import math

# Written 11/1/2016
# Label Line features
# Pre-condition: Gets the depth image and list of line segments and parameters
# Post-condition: Returns a list of lines labeled as curvature edges or discontinuities
# ----------------
# We define curvature edges as lines (or curves) on the object and discontinuities as those lines
# on the outside edges of the object where you can see it touching the background.


def get_orientation(line, mask_size):
    startpt = [line[0], line[1]]
    endpt = [line[2], line[3]]
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])

    # determine if the line is vertical or horizontal
    # and set up the coords for the region of interest
    if dy > dx or dy == dx:
        # Vertical line
        pt1 = [line[0], line[1] - mask_size]
        pt2 = [line[0], line[1] + mask_size]
        pt3 = [line[2], line[3] - mask_size]
        pt4 = [line[2], line[3] + mask_size]
        # line.append(1)	# tag for vertical lines
        # we'll add this after labeling the discontinuity lines
        win_pos = [startpt, endpt, pt4, pt2]
        win_neg = [pt1, pt3, endpt, startpt]
    else:
        # Horizontal line
        pt1 = [line[0] - mask_size, line[1]]
        pt2 = [line[0] + mask_size, line[1]]
        pt3 = [line[2] - mask_size, line[3]]
        pt4 = [line[2] + mask_size, line[3]]
        line.append(2)  # horizontal line
        win_pos = [startpt, pt4, endpt, pt2]
        win_neg = [pt1, endpt, pt3, startpt]
    window = [pt1, pt2, pt3, pt4]
    return window, win_pos, win_neg


def label_line_features(depthimg, seglist, parameters):
    # Constants
    minlen = int(parameters["Cons_Lmin"])
    dis_thresh = int(parameters["thresh_label_dis"])
    mask_size = int(parameters["label_win_sized"])

    # Get the lines which are longer than the minimum length
    desired_lines = [line for line in seglist if line[4] > minlen]

    for line in desired_lines:
        window, win_pos, win_neg = get_orientation(line, mask_size)
        mask = cv2.boundingRect(window)