from math import inf, sqrt, atan, degrees
import numpy as np
import cv2

from utility import normalize_depth


def calc_inf(y2, y1, x2, x1):
    return inf if (y2 - y1) > 0 else -inf


def find_star(x, y, idx, ListEdges):
    # print('x', x, 'y', y, ListEdges[idx])
    stx = np.where(ListEdges[idx][:, 0][:, 0] == x)
    sty = np.where(ListEdges[idx][:, 0][:, 1] == y)
    # Get the first element of the single-item set
    return next(iter(set(sty[0]).intersection(stx[0])))


def get_lin_index(x1, y1, img, imgsize):
    # print('x1', x1, 'y1', y1)
    img = normalize_depth(img, colormap=True)
    # cv2.line(img, (600, 100), (600, 100), (0, 0, 255), 3)
    # cv2.line(img, (x1, y1), (x1, y1), (0, 0, 255), 3)
    # cv2.namedWindow('Points', cv2.WINDOW_NORMAL)
    # cv2.imshow('Points', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return np.ravel_multi_index((x1, y1), imgsize, order='C')


def create_linefeatures(ListSegments, ListEdges, img, imgsize):
    curr_idx = 0
    LineFeature = []
    ListPoint = []

    for i, curr in enumerate(ListSegments):
        for j in range(curr.shape[0] - 1):
            # print(i, '.', curr)
            x1, y1 = curr[j].astype(int)
            x2, y2 = curr[j + 1].astype(int)
            # print('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2)


            slope = round((y2 - y1) / (x2 - x1), 4) if ((x2 - x1) != 0) else calc_inf(y2, y1, x2, x1)
            lin_ind1 = get_lin_index(x1, y1, img, imgsize)
            lin_ind2 = get_lin_index(x2, y2, img, imgsize)
            # print('linear indices:', lin_ind1, lin_ind2)
            linelen = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            alpha = degrees(atan(-slope))

            LineFeature.append([x1, y1, x2, y2, linelen, slope, alpha, curr_idx, lin_ind1, lin_ind2])
            curr_idx += 1

            a = find_star(x1, y1, i, ListEdges)
            b = find_star(x2, y2, i, ListEdges)
            ListPoint.append(ListEdges[i][:, 0][a:b + 1])

            if LineFeature[curr_idx - 2][8: 10] == [lin_ind1, lin_ind2] and curr_idx > 2:
                del (LineFeature[curr_idx - 1])
                del (ListPoint[curr_idx - 1])
                curr_idx -= 1

    len_lp = len(ListPoint)
    LPP = []
    for cnt in range(len_lp):
        LPP.append([np.ravel_multi_index((ListPoint[cnt][:, 0], ListPoint[cnt][:, 1]), imgsize, order='C')])

    return np.array(LineFeature), np.array(LPP)



