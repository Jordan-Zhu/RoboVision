from math import inf, sqrt, atan, degrees
import numpy as np


def calc_inf(y2, y1, x2, x1):
    return inf if (y2 - y1) > 0 else -inf


def find_star(x, y, idx, ListEdges):
    # print('x', x, 'y', y, ListEdges[idx])
    # print(ListEdges[idx][:, 0])
    sty = np.where(ListEdges[idx][:, 0] == y)
    stx = np.where(ListEdges[idx][:, 1] == x)
    # print(sty, stx)
    # Get the first element of the single-item set
    return next(iter(set(sty[0]).intersection(stx[0])))


def get_lin_index(x1, y1, imgsize):
    return np.ravel_multi_index((y1, x1), imgsize, order='F')


def create_linefeatures(ListSegments, ListEdges, imgsize):
    c0 = 0
    LineFeature = []
    ListPoint = []

    for i, curr in enumerate(ListSegments):
        for j in range(curr.shape[0] - 1):
            # print('i', i, 'j', j)
            # print('curr', curr)
            x1, y1 = curr[j].astype(int)
            x2, y2 = curr[j + 1].astype(int)
            # print('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2)

            slope = round((y2 - y1) / (x2 - x1), 4) if ((x2 - x1) != 0) else calc_inf(y2, y1, x2, x1)
            lin_ind1 = get_lin_index(x1, y1, imgsize)
            lin_ind2 = get_lin_index(x2, y2, imgsize)
            linelen = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            alpha = degrees(atan(-slope))

            LineFeature.append([y1, x1, y2, x2, linelen, slope, alpha, c0, lin_ind1, lin_ind2])
            c0 += 1

            a = find_star(x1, y1, i, ListEdges)
            b = find_star(x2, y2, i, ListEdges)
            ListPoint.append(ListEdges[i][a:b + 1])

            if LineFeature[c0 - 2][8: 10] == [lin_ind1, lin_ind2] and c0 > 2:
                del (LineFeature[c0 - 1])
                del (ListPoint[c0 - 1])
                # print('Duplicate removed')
                c0 -= 1

    len_lp = len(ListPoint)
    LPP = []
    for cnt in range(len_lp):
        LPP.append(np.ravel_multi_index((ListPoint[cnt][:, 0], ListPoint[cnt][:, 1]), imgsize, order='F'))

    return np.array(LineFeature), np.array(LPP)
