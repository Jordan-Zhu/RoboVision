import cv2
import numpy as np
from utility import *


def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)

    overlay = normalize_depth(src, colormap=True)
    output = normalize_depth(src, colormap=True)
    alpha = 0.5

    win = swap_indices(poly)
    # print(win)

    cv2.fillConvexPoly(mask, win, 255)  # Create the ROI
    cv2.fillConvexPoly(overlay, win, (255, 255, 255))
    cv2.putText(overlay, "ROI Poly: alpha={}".format(alpha), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    res = src * mask
    # print('mask1 count', np.count_nonzero(mask))
    # cv2.namedWindow('roi poly', cv2.WINDOW_NORMAL)
    # cv2.imshow('roi poly', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res


def classify_curves(src, list_lines, list_points, window_size):
    im_size = src.shape
    img = normalize_depth(src, colormap=True)
    res = []
    for index, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4 = get_orientation(line, window_size)

        cv2.line(img, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), (0, 0, 255), 1)
        cv2.line(img, (int(pt3[1]), int(pt3[0])), (int(pt4[1]), int(pt4[0])), (0, 0, 255), 1)
        cv2.line(img, (int(line[1]), int(line[0])), (int(line[3]), int(line[2])), (0, 255, 0), 1)

        win = np.array(get_ordering(pt1, pt2, pt3, pt4))
        mask4 = roipoly(src, win)

        # Get mask of values on the line
        lx = [list_points[index]]
        # print(lx)
        temp_list = []
        for ii in lx:
            r1, c1 = np.unravel_index([ii], im_size, order='F')
            # print(x1, ",", y1)
            temp_list.append([c1[0], r1[0]])

        mask5 = []
        for i in temp_list:
            # print(i[1], i[0])
            mask5.append(src[i[1], i[0]])
        mask5 = np.array(mask5)
        # print('mask5', mask5)
        # Eliminate zeros
        mask5 = mask5[np.nonzero(mask5)]

        # Average values in the mask
        mask4_size = cv2.countNonZero(mask4)
        a1 = 0 if cv2.countNonZero(mask4) == 0 else sum(src[np.nonzero(mask4)]) / mask4_size

        # print('mask5', np.mean(mask5))
        mask5_size = mask5.shape[0]
        a2 = 0 if cv2.countNonZero(mask5) == 0 else np.mean(mask5)

        # print('A1', a1, '\nA2', a2, '\n')

        b1 = mask4_size * a1 - mask5_size * a2
        try:
            b11 = float(b1) / (cv2.countNonZero(mask4) - len(mask5))
        except ZeroDivisionError:
            b11 = float('nan')

        # print('B11', b11)
        if b11 < a2:
            res.append(np.append(list_lines[index], [12]))
        else:
            res.append(np.append(list_lines[index], [13]))

    cv2.namedWindow('Points', cv2.WINDOW_NORMAL)
    cv2.imshow('Points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.asarray(res)
