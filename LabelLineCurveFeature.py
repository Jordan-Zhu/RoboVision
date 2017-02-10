import cv2
import numpy as np
import numpy.ma as ma
import matplotlib as plt

from utility import get_orientation, get_ordering, normalize_depth

# Written 11/8/2016
# Label Line curve feature
# Pre-condition: Gets the depth image and list of line segments and parameters
# Post-condition: Returns a list of lines labeled as curvature edges or discontinuities
# ----------------
# We define curvature edges as lines (or curves) on the object and discontinuities as those lines
# on the outside edges of the object where you can see it touching the background.

# 12/5/2016 - Gets some of the lines, but not all and doesn't get any lines past a particular index


def swap_indices(arr):
    res = []
    for i, e in enumerate(arr):
        res.append([arr[i][1], arr[i][0]])
    return np.array(res)


def roipoly(src, poly):
    # img = normalize_depth(src, colormap=True)
    mask = np.zeros_like(src, dtype=np.uint8)

    cv2.fillConvexPoly(mask, poly, 255) # Create the ROI
    # cv2.fillPoly(mask, np.int32([poly]), 255)
    # cv2.drawContours(mask, np.int32([poly]), -1, 255, -1)
    res = src * mask
    print('mask1 count', np.count_nonzero(mask))
    # cv2.namedWindow('roi poly', cv2.WINDOW_NORMAL)
    # cv2.imshow('roi poly', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res


def roipoly2(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)

    for i in range(poly[0][0], poly[3][0]):
        a = 1



def classify_curves(src, list_lines, list_points, window_size):
    im_size = src.shape
    out = []
    for index, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4 = get_orientation(line, window_size)
        win = get_ordering(pt1, pt2, pt3, pt4)
        win = swap_indices(win)

        print(win)
        mask4 = roipoly(src, win)
        area = cv2.contourArea(np.int32([win]))
        print('Mask4 area:', area)
        print('Mask4 sum:', sum(src[np.nonzero(mask4)]))
        # print('Mask4 shape:', mask4[np.nonzero(mask4)].shape)
        # cv2.countNonZero(cv2.cvtColor(mask4, cv2.COLOR_BGR2GRAY)

        # a1 = np.mean(mask4[np.nonzero(mask4)])
        a1 = sum(src[np.nonzero(mask4)]) / cv2.countNonZero(mask4)
        lx = list_points[index]
        # print('lx', lx)
        temp_list = []
        for ii in lx:
            t1, t2 = np.unravel_index([ii - 1], im_size, order='F')
            temp_list.append([t1[0], t2[0]])
        # print('Temp list:', temp_list)

        mask5 = []
        for i in temp_list:
            mask5.append(src[i[0], i[1]])
        # print('Mask 5:', mask5)


        mask5 = [value for value in mask5 if value != 0]
        a2 = np.mean(mask5)
        print('A1', a1, '\nA2', a2)

        b1 = len(mask4) * a1 - len(mask5) * a2
        # b1 = np.count_nonzero(mask4) * a1 - len(mask5) * a2
        try:
            b11 = float(b1) / (len(mask4) - len(mask5))
        except ZeroDivisionError:
            b11 = float('nan')

        # print('b11', b11, '\n')
        if b11 < a2:
            print('IF\n\n')
            out.append(np.append(list_lines[index], [12]))
        else:
            print('ELSE\n\n')
            out.append(np.append(list_lines[index], [13]))

    return np.asarray(out)
