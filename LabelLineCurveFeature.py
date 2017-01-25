import cv2
import numpy as np

from utility import get_orientation, get_ordering, normalize_depth

# Written 11/8/2016
# Label Line curve feature
# Pre-condition: Gets the depth image and list of line segments and parameters
# Post-condition: Returns a list of lines labeled as curvature edges or discontinuities
# ----------------
# We define curvature edges as lines (or curves) on the object and discontinuities as those lines
# on the outside edges of the object where you can see it touching the background.

# 12/5/2016 - Gets some of the lines, but not all and doesn't get any lines past a particular index


def roipoly(src, line, poly):
    src = normalize_depth(src, colormap=True)
    mask = np.zeros_like(src, dtype=np.uint8)
    # print("poly:", poly)
    cv2.rectangle(mask, (poly[0][1], poly[0][0]), (poly[3][1], poly[3][0]), (255, 255, 255), cv2.FILLED)
    res = src * mask
    # cv2.imshow("image", res)
    # cv2.waitKey(0)
    return res



def classify_curves(src, list_lines, list_points, window_size):
    im_size = src.shape
    out = []
    # print("List lines:", list_lines.shape)
    for index, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4 = get_orientation(line, window_size)
        win = get_ordering(pt1, pt2, pt3, pt4)
        poly = [pt1, pt2, pt3, pt4]

        win = [[int(i) for i in pt] for pt in poly]
        # Use numpy mask function to create masked array
        mask4 = roipoly(src, line, win)
        # print('mask4:', np.nonzero(mask4))
        # mask4 = [value for value in mask4 if value != 0]

        # a1 = np.mean(mask4)
        a1 = sum(mask4) / len(mask4)
        print(len(mask4))
        # a1 = np.mean(mask4[np.nonzero(mask4)])
        lx = list_points[index]
        # print("lx:", lx[0][0])
        temp_list = []
        for ii in lx[0]:
            t1, t2 = np.unravel_index(ii, im_size, order='F')
            # print(t1[0], ",", t2[0])
            # temp4 = [temp3[0][0], temp3[1][0]]
            # print(temp4)
            temp_list.append([t1[0], t2[0]])
        # print(temp_list)

        mask5 = []
        # print(src[100, 100])
        for i in temp_list:
            mask5.append(src[i[0], i[1]])
        # print("mask5:", mask5)


        mask5 = [value for value in mask5 if value != 0]
        a2 = np.mean(mask5)
        b1 = len(mask4) * a1 - len(mask5) * a2
        try:
            b11 = float(b1) / (len(mask4) - len(mask5))
        except ZeroDivisionError:
            b11 = float('nan')

        if b11 < a2:
            out.append(np.append(list_lines[index], [12]))
            # print("Negative line:", index)
        else:
            out.append(np.append(list_lines[index], [13]))
            # print("Positive line:", index)

    # check_output('linenewout.mat', np.asarray(out), 'Line_newC')
    return np.asarray(out)
