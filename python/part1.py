# Import the necessary packages
import cv2
import scipy.io as sio
import numpy as np
from utility import *
from lineseg import lineseg
from drawedgelist import drawedgelist
from python.Lseg_to_Lfeat_v4 import create_linefeatures
from python.merge_lines_v4 import merge_lines
from python.LabelLineCurveFeature_v4 import classify_curves


def initContours(img):
    edges = edge_detect(img)
    cntrs = np.asarray(find_contours(edges))

    squeeze_ndarr(cntrs)

    seg_list = lineseg(cntrs, tol=2)
    # ADVANCED SLICING
    for i in range(seg_list.shape[0]):
        swap_cols(seg_list[i], 0, 1)
    for i in range(cntrs.shape[0]):
        swap_cols(cntrs[i], 0, 1)
    return seg_list, cntrs


def draw_lfeat(line_feature, img):
    blank_image = normalize_depth(img, colormap=True)
    # blank_image = np.zeros_like(img)

    for i, e in enumerate(line_feature):
        x1 = int(e[1])
        y1 = int(e[0])
        x2 = int(e[3])
        y2 = int(e[2])
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.line(blank_image, (x1, y1), (x2, y2), color, 3)

    cv2.namedWindow('Line features', cv2.WINDOW_NORMAL)
    cv2.imshow('Line features', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('learn0.png', -1)
    im_size = img.shape
    seg_list, cntrs = initContours(img)

    LineFeature, ListPoint = create_linefeatures(seg_list, cntrs, im_size)
    Line_new, ListPoint_new, line_merged = merge_lines(LineFeature, ListPoint, 10, im_size)
    draw_lfeat(Line_new, img)
    line_newC = classify_curves(img, Line_new, ListPoint_new, 11)
    draw_convex(line_newC, img)

