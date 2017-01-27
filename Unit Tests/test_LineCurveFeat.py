import cv2
import scipy.io as sio
import numpy as np

from merge_lines_v3 import merge_lines
from LabelLineCurveFeature import classify_curves
from utility import normalize_depth


def roipoly(src, line, poly):
    mask = np.zeros_like(src)
    dst = np.zeros_like(src)
    cv2.rectangle(mask, (poly[0][1], poly[0][0]), (poly[3][1], poly[3][0]), (255, 255, 255), cv2.FILLED)
    cv2.bitwise_and(src, dst, mask=mask)
    cv2.imshow("image", dst)


if __name__ == '__main__':
    depthimg = cv2.imread('learn0.png', -1)
    src = normalize_depth(depthimg, colormap=True)
    poly = [[100, 100], [0, 0], [0,0], [200, 200]]

    siz = depthimg.shape
    thresh_m = 10
    label_thresh = 11

    # mask = np.zeros_like(src, dtype=np.uint8)
    # dst = np.zeros_like(src, dtype=np.uint8)
    # print("mask shape:", mask.shape, "src shape:", src.shape)
    # cv2.rectangle(mask, (poly[0][1], poly[0][0]), (poly[3][1], poly[3][0]), (255, 255, 255), cv2.FILLED)
    # res = cv2.bitwise_and(src, src, mask=mask)
    # final_im = mask * src
    # final = cv2.bitwise_or(src, dst)
    # res = np.copyto(dst, src, where=)
    # cv2.imshow("image", src)
    # cv2.waitKey(0)

    data = sio.loadmat('input_LLCF_1.mat')
    out_mat = sio.loadmat('out_LLCF_1.mat')
    # data = sio.loadmat('LabelLineCurveFeature_v2.mat')
    # data2 = sio.loadmat('Id.mat')
    # data2 = sio.loadmat('Parameter.mat')
    #
    # # inputs
    # print('Id:', data2)
    # Id = data['Id']
    Line_newC = data['Line_newC']
    ListPoint_newC = data['ListPoint_newC']
    out_Line_new = out_mat['Line_newC']
    #
    # Parameter = data2['P']

    # [line_new, listpoint_new, line_merged] = merge_lines(Line_newC, ListPoint_newC, thresh_m, siz)
    line_new = classify_curves(depthimg, Line_newC, ListPoint_newC, label_thresh)
    # print(line_new[:, 10])
    # print(out_Line_new[:, 10])

    # Checking the output is the same as matlab
    sum = 0
    for i in range(len(line_new)):
        if line_new[i, 10] == out_Line_new[i, 10]:
            print(i, ".", int(line_new[i, 10]), " == ", int(out_Line_new[i, 10]), "\n")
            sum += 1
    print('Total lines:', len(line_new), ' Total correct lines:', sum)
    # print(*line_new[:, 10], sep='\n')
