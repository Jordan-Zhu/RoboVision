import cv2
import scipy.io as sio
from LabelLineCurveFeature import classify_curves

from legacyCode.utility import normalize_depth


def squeeze_array(arr):
    res = []
    for i in range(arr.shape[0]):
        res.append(arr[i][0])
    return res


if __name__ == '__main__':
    depthimg = cv2.imread('learn0.png', -1)
    src = normalize_depth(depthimg, colormap=True)
    poly = [[100, 100], [0, 0], [0,0], [200, 200]]

    siz = depthimg.shape
    thresh_m = 10
    label_thresh = 11

    data = sio.loadmat('LLCF_1.mat')
    Line_newC = data['Line_newC']
    Line_newCx = data['Line_newCx']
    ListPoint_newC = squeeze_array(data['ListPoint_newC'])

    # print(ListPoint_newC[0])

    line_new = classify_curves(depthimg, Line_newC, ListPoint_newC, label_thresh)

    # Checking the output is the same as matlab
    # sum = 0
    # for i in range(len(line_new)):
    #     if line_new[i, 10] == out_Line_new[i, 10]:
    #         print(i, ".", int(line_new[i, 10]), " == ", int(out_Line_new[i, 10]), "\n")
    #         sum += 1
    # print('Total lines:', len(line_new), ' Total correct lines:', sum)

    # print(line_new[:, 10])
    # print(Line_newCx[:, 10])
