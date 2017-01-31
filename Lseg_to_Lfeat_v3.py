from math import inf, sqrt, atan, degrees
import numpy as np


def calc_inf(y2, y1, x2, x1):
    return inf if (y2 - y1) > 0 else -inf


def find_star(x, y, idx, ListEdges):
    # print(ListEdges[0][idx][:, 0])
    # print('x = ', x, ' y = ', y, sep='')
    sty = np.where(ListEdges[0][idx][:, 0] == y)
    stx = np.where(ListEdges[0][idx][:, 1] == x)
    # print('sty = ', sty, ' stx = ', stx, sep='')
    # Get the first element of the single-item set
    return next(iter(set(sty[0]).intersection(stx[0])))
    # return set(sty[0]).intersection(stx[0])


def create_linefeatures(ListSegments, ListEdges, imgsize):
    # print('imgsize', imgsize)
    # print(np.ravel_multi_index((310, 34), imgsize, order='F')+1)
    # print(ListSegments[0])
    # length = ListSegments.shape[1]
    # for i in range(length):
    # print(ListSegments[0, 1].shape[0])

    # print(ListEdges[0, 1])
    # print(ListEdges[0][1][:, 0])
    # print(ListEdges[0][1][:, 1])
    # print("\n")
    # print(ListEdges.shape)
    c0 = 0
    dup = 0
    LineFeature = []
    ListPoint = []

    for i, curr in enumerate(ListSegments[0]):
        # print(curr)
        for j in range(0, curr.shape[0] - 1):
            y1, x1 = curr[j].astype(int)
            y2, x2 = curr[j + 1].astype(int)

            # print("y1:", y1, "y2:", y2, "x1:", x1, "x2:", x2)
            slope = round((y2 - y1) / (x2 - x1), 4) if ((x2 - x1) != 0) else calc_inf(y2, y1, x2, x1)
            # print("slope", slope, "\n")
            lin_ind1 = np.ravel_multi_index((y1 - 1, x1 - 1), imgsize, order='F') + 1
            lin_ind2 = np.ravel_multi_index((y2 - 1, x2 - 1), imgsize, order='F') + 1
            # print("lind1:", lin_ind1, ",lind2:", lin_ind2)
            linelen = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            alpha = degrees(atan(-slope))

            LineFeature.append([y1, x1, y2, x2, linelen, slope, alpha, c0, lin_ind1, lin_ind2])
            # print(LineFeature[c0])
            c0 += 1

            a = find_star(x1, y1, i, ListEdges)
            b = find_star(x2, y2, i, ListEdges)
            # print('a = ', a, ', b = ', b, '\n', sep='')
            # s = next(iter(a))
            # e = next(iter(b))
            # print('s = ', s, ', e = ', e, sep='')
            # print(ListEdges[0][c0][s:e])
            ListPoint.append(ListEdges[0][i][a:b + 1])
            # print(ListPoint[i])

            # print("Linefeat:", LineFeature[c0 - 2][8: 10])
            if LineFeature[c0 - 2][8: 10] == [lin_ind1, lin_ind2] and c0 > 2:
                # print("=============== Duplicate found ===================")
                # print("c0 =", c0, "\n")
                # print(LineFeature[c0 - 2])
                # print(LineFeature[c0 - 1])
                # print(ListPoint[c0 - 2])
                # print(ListPoint[c0 - 1])
                del (LineFeature[c0 - 1])
                del (ListPoint[c0 - 1])
                c0 -= 1
                dup += 1

    len_lp = len(ListPoint)
    # print(ListPoint[0][:, 0])
    print(len_lp)
    LPP = []
    for cnt in range(len_lp):
        LPP.append([np.ravel_multi_index((ListPoint[cnt][:, 0] - 1, ListPoint[cnt][:, 1] - 1), imgsize, order='F') + 1])
    print(LPP[0])

            # c0 += 1
    # print("Duplicate lines:", dup)
    # print(LineFeature[554])
    return LineFeature, LPP
