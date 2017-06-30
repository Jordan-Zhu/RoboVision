import numpy as np

from utility import roipoly, get_orientation, get_ordering

def classify_curves(src, list_lines, list_points, window_size):
    im_size = src.shape

    out = []
    for index, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4 = get_orientation(line, window_size)
        win = get_ordering(pt1, pt2, pt3, pt4)
