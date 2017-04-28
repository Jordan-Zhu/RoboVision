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
from python.LabelLineFeature_v1 import label_line_features
from python.line_match import line_match


def initContours(img):
    # edges = edge_detect(img)
    curve_disc, curve_con, depth_disc, depth_con, edges = edge_detect(img)

    seg_list = lineseg(edges, tol=2)
    cntrs = find_contours(img)
    # ADVANCED SLICING
    for i in range(cntrs.shape[0]):
        swap_cols(cntrs[i], 0, 1)
    return seg_list, edges, cntrs


def draw_lfeat(line_feature, img):
    # blank_image = normalize_depth(img, colormap=True)
    height = img.shape[0]
    width = img.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)

    for i, e in enumerate(line_feature):
        x1 = int(e[1])
        y1 = int(e[0])
        x2 = int(e[3])
        y2 = int(e[2])
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.line(blank_image, (x1, y1), (x2, y2), color, 1)

    cv2.namedWindow('Line features', cv2.WINDOW_NORMAL)
    cv2.imshow('Line features', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_listpair(list_pair, line_feature, img):
    blank_image = normalize_depth(img, colormap=True)

    for i, e in enumerate(list_pair):
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        for j, e in enumerate(e):
            line = line_feature[np.where(line_feature[:, 7] == e)[0]][0]
            x1 = int(line[1])
            y1 = int(line[0])
            x2 = int(line[3])
            y2 = int(line[2])
            cv2.line(blank_image, (x1, y1), (x2, y2), color, 2)

    cv2.namedWindow('Line features', cv2.WINDOW_NORMAL)
    cv2.imshow('Line features', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('../img/learn0.png', -1)
    # img = img[50:, 50:480]
    im_size = img.shape

    P = sio.loadmat('Parameter.mat')
    param = P['P']

    curve_disc, curve_con, depth_disc, depth_con, dst = edge_detect(img)
    height = img.shape[0]
    width = img.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)
    # draw_contours(blank_image, dst)
    # drawedgelist(dst)

    # print(dst)

    # Remove extra dimensions from data
    res = lineseg(dst, tol=2)
    seglist = []
    for i in range(res.shape[0]):
        # print('shape', res[i].shape)
        if res[i].shape[0] > 2:
            # print(res[i])
            # print(res[i][0])
            seglist.append(np.concatenate((res[i], [res[i][0]])))
        else:
            seglist.append(res[i])

    seglist = np.array(seglist)

    # print(seglist)

    # seg_curve = lineseg(curve_con, tol=1)
    # seg_disc = lineseg(depth_con, tol=1)
    # seg_list = np.hstack((seg_curve, seg_disc))

    # print(seg_disc)
    # seg_list, edges, cntrs = initContours(img)
    # print(dst.shape)

    drawedgelist(seglist)

    # drawedgelist(seg_curve)

    LineFeature, ListPoint = create_linefeatures(seglist, dst, im_size)
    Line_new, ListPoint_new, line_merged = merge_lines(LineFeature, ListPoint, 10, im_size)
    draw_lfeat(Line_new, img)
    # print(line_merged)

    # line_newC = classify_curves(img, Line_new, ListPoint_new, 11)
    # draw_convex(line_newC, img)

    # LineFeature_curve, ListPoint_curve = create_linefeatures(seg_curve, curve_con, im_size)
    # Line_new, ListPoint_new, line_merged = merge_lines(LineFeature_curve, ListPoint_curve, 10, im_size)
    # print('Line_new size:', Line_new.shape)
    # draw_lfeat(Line_new, img)
    #
    # LineFeature_disc, ListPoint_disc = create_linefeatures(seg_disc, depth_con, im_size)
    # Line_new, ListPoint_new, line_merged = merge_lines(LineFeature_disc, ListPoint_disc, 10, im_size)
    # print('Line_new size:', Line_new.shape)
    # draw_lfeat(Line_new, img)

    # seg_list, edges, cntrs = initContours(img)
    #
    # LineFeature, ListPoint = create_linefeatures(seg_list, cntrs, im_size)
    # Line_new, ListPoint_new, line_merged = merge_lines(LineFeature, ListPoint, 10, im_size)
    # draw_lfeat(Line_new, img)
    # line_newC = classify_curves(img, Line_new, ListPoint_new, 11)
    # draw_convex(line_newC, img)
    #
    # # Remove the 11th column for post-processing
    # line_newC = np.delete(line_newC, 10, axis=1)
    # line_new_new = label_line_features(img, edges, line_newC, param)
    # print('Line_new:', line_new_new.shape)
    #
    # # Keep the lines that are curvature / discontinuities
    # relevant_lines = np.where(line_new_new[:, 10] != 0)[0]
    # line_interesting = line_new_new[relevant_lines]
    # # Fast sorting, done based on line angle
    # line_interesting = line_interesting[line_interesting[:, 6].argsort()]
    #
    # print('Line interesting:', line_interesting.shape)
    # draw_lfeat(line_interesting, img)
    #
    # # Match the lines into pairs
    # list_pair = line_match(line_interesting, param)
    # print('List pair:', list_pair)
    # draw_listpair(list_pair, line_interesting, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



