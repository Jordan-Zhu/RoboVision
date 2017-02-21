import cv2
import numpy as np
from utility import get_orientation, get_ordering, normalize_depth


def swap_indices(arr):
    res = []
    for i, e in enumerate(arr):
        res.append([arr[i][1] - 1, arr[i][0] - 1])
    return np.array(res)


def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)

    cv2.fillConvexPoly(mask, poly, 255) # Create the ROI
    res = src * mask
    # print('mask1 count', np.count_nonzero(mask))
    # cv2.namedWindow('roi poly', cv2.WINDOW_NORMAL)
    # cv2.imshow('roi poly', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res


def classify_curves(src, list_lines, list_points, window_size):
    im_size = src.shape
    out = []
    for index, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4 = get_orientation(line, window_size)
        win = get_ordering(pt1, pt2, pt3, pt4)
        # win2 = get_ordering(pt1, pt2, pt3, pt4)
        win = swap_indices(win)

        # print(win)
        mask4 = roipoly(src, win)
        # mask0 = np.array(roipoly2(src, line, win2))
        area = cv2.contourArea(np.int32([win]))
        # print('Mask4 area:', area)
        # print('Mask4 count:', cv2.countNonZero(mask4))


        # a1 = np.mean(mask4[np.nonzero(mask4)])
        a1 = 0 if cv2.countNonZero(mask4) == 0 else sum(src[np.nonzero(mask4)]) / cv2.countNonZero(mask4)
        # a1 = sum(mask4)) / cv2.countNonZero(np.array(mask4))

        # Get mask of values on the line
        lx = list_points[index]
        temp_list = []
        for ii in lx:
            t1, t2 = np.unravel_index([ii - 1], im_size, order='F')
            temp_list.append([t1[0], t2[0]])

        mask5 = []
        for i in temp_list:
            mask5.append(src[i[0], i[1]])

        mask5 = [value for value in mask5 if value != 0]
        # print('mask5', mask5)
        a2 = np.mean(mask5)

        print('A1', a1, '\nA2', a2, '\n')

        b1 = cv2.countNonZero(mask4) * a1 - len(mask5) * a2
        try:
            b11 = float(b1) / (cv2.countNonZero(mask4) - len(mask5))
        except ZeroDivisionError:
            b11 = float('nan')

        # print('B11', b11)
        if b11 < a2:
            # print('IF\n\n')
            out.append(np.append(list_lines[index], [12]))
        else:
            # print('ELSE\n\n')
            out.append(np.append(list_lines[index], [13]))

    return np.asarray(out)
