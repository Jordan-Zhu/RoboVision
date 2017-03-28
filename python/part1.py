# Import the necessary packages
import cv2
import scipy.io as sio
import numpy as np
from utility import *
from lineseg import lineseg
from drawedgelist import drawedgelist
from python.Lseg_to_Lfeat_v4 import create_linefeatures


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

if __name__ == '__main__':
    img = cv2.imread('learn0.png', -1)
    seg_list, cntrs = initContours(img)

    LineFeature, ListPoint = create_linefeatures(seg_list, cntrs, img.shape)

