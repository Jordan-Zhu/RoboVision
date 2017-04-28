# Robot Grasping Algorithm

# Importing the necessary packages:
from cv2 import *
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from skimage import morphology

from utility import showimg, normalize_depth, edge_detect, find_contours
from drawlinefeature import DrawLineFeature,drawconvex
from lineseg import lineseg
from drawedgelist import drawedgelist
from Lseg_to_Lfeat_v2 import Lseg_to_Lfeat_v2
from LabelLineCurveFeature_v2 import LabelLineCurveFeature_v2
from merge_lines_v2 import merge_lines
from LabelLineCurveFeature import classify_curves


if __name__ == '__main__':
    thresh_m = 10
    label_thresh = 11

    # Load the depth image with alpha channel, hence -1
    depthimg = cv2.imread('img/learn0.png', -1)
    colorimg = cv2.imread('img/clearn17.png', 0)

    showimg(normalize_depth(depthimg, colormap=True), 'Depth img')

    id = depthimg[100:, 100:480]  ## zc crop the region of interest
    siz = id.shape  ## image size of the region of interest

    # Find edges in the depth image as gradient plus the depth image in range (0 - 255)
    edges = edge_detect(id)
    showimg(edges, "Canny of depth image + discontinuity")

    # Next, find the contours
    cntrs = np.asarray(find_contours(edges))

    # Create line segments from the contours
    seglist = lineseg(cntrs, tol=2)

    print("seglist shape:", len(seglist))
    print(seglist, sep='\n')

    drawedgelist(seglist, rowscols=[])

    # Get line features (slope, length, angle, linear indices) for later steps
    [linefeature, listpoint] = Lseg_to_Lfeat_v2(seglist, cntrs, siz)
    # Merge visually similar lines
    [line_new, listpoint_new, line_merged] = merge_lines(linefeature, listpoint, thresh_m, siz)

    # line_new = LabelLineCurveFeature_v2(depthimg, line_new, listpoint_new, label_thresh)

    # Label curvatures as either convex or concave
    line_new = classify_curves(depthimg, line_new, listpoint_new, label_thresh)

    # line_new = LabelLineCurveFeature_v2(depthimg, line_new, listpoint_new, label_thresh)

    DrawLineFeature(linefeature, siz, 'line features')
    drawconvex(line_new, siz, 'convex')

    # TO-DO
    # - Check LabelLineCurveFeature_v2 with python input
    # - Write a function to make a window mask
    # - (Section 4) Take non-zero curve features and sort by angle (index 7 in MATLAB)

    # Long-term, to make the program easier to read and use
    # *- Create a Line object with properties: start, end, object/background, curvature/discontinuity
    #   distance from another line, check if line is overlapping
