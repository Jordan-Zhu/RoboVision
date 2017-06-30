import math
import numpy as np
from itertools import combinations, chain
from collections import Counter


def compare(s, t):
    return Counter(s) == Counter(t)


def math_stuff(x1, y1, x2, y2):
    slope = float((y2 - y1) / (x2 - x1) if ((x2 - x1) != 0) else math.inf)
    line_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = math.degrees(math.atan(-slope))
    return slope, line_len, alpha


def relevant_lines(i, pairs, lines):
    pt1 = pairs[i][0]
    pt2 = pairs[i][1]
    line1 = lines[pt1]
    line2 = lines[pt2]
    alph1 = line1[6]
    alph2 = line2[6]
    temp1 = [line1[8], line1[9]]
    temp2 = [line2[8], line2[9]]
    return pt1, pt2, alph1, alph2, temp1, temp2


def merge_listpoints(listpt, pt1, pt2, px1, px2):
    lp1 = listpt[pt1]
    lp2 = listpt[pt2]
    startpt1 = np.where(lp1 == px1)[0]
    startpt2 = np.where(lp1 == px2)[0]
    startpt3 = np.where(lp2 == px1)[0]
    startpt4 = np.where(lp2 == px2)[0]

    if not startpt1 and startpt1.shape[0] < 1:
        line_start = lp2
        line_end = lp1

        if startpt3 > 0:
            line_start = line_start[::-1]
        if startpt2 == 0:
            line_end = line_end[::-1]
    else:
        line_start = lp1
        line_end = lp2

        if startpt1 > 0:
            line_start = line_start[::-1]
        if startpt4 == 0:
            line_end = line_end[::-1]

    # print(listpt[max(pt1, pt2)])
    # listpt = np.delete(listpt, max(pt1, pt2), axis=0)
    # listpt = np.delete(listpt, min(pt1, pt2), axis=0)
    del listpt[max(pt1, pt2)]
    del listpt[min(pt1, pt2)]
    merged = np.r_[line_start[0:-1], line_end]
    # print('merged', merged)
    listpt.append(merged)

    return listpt


def merge_lines(lines, listpt, thresh, imgsize):
    listpt = list(listpt)
    out = [[n] for n in range(0, lines.shape[0])]
    unique_pts = np.sort(np.unique(lines[:, 8:10]))

    for index, ptx in enumerate(unique_pts):
        pairs = list(combinations(list(np.where(lines == ptx)[0]), 2))
        if not pairs:
            continue
        for i in range(len(pairs)):
            pt1, pt2, alph1, alph2, temp1, temp2 = relevant_lines(i, pairs, lines)
            # Check that the lines are within the threshold and not coincident
            if abs(alph1 - alph2) > thresh or compare(temp1, temp2):
                continue

            lind1, lind2 = np.sort([int(i) for i in list(filter(lambda e: e not in [ptx], chain(temp1 + temp2)))])
            # print('linear indices: ', lind1, lind2)
            x1, y1 = np.squeeze(np.unravel_index([lind1], imgsize, order='C'))
            x2, y2 = np.squeeze(np.unravel_index([lind2], imgsize, order='C'))
            # print('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2)
            slope, line_len, alpha = math_stuff(x1, y1, x2, y2)

            # Intersection point is in the middle of the new line
            if min(alph1, alph2) <= alpha <= max(alph1, alph2):
                lines = np.delete(lines, max(pt1, pt2), axis=0)
                lines = np.delete(lines, min(pt1, pt2), axis=0)
                val1 = out[pt1]
                val2 = out[pt2]
                del out[max(pt1, pt2)]
                del out[min(pt1, pt2)]

                # Update both lists to reflect the addition of the merged line.
                lines = np.append(lines, [[int(x1), int(y1), int(x2), int(y2), line_len, slope, alpha, 0, lind1, lind2]], axis=0)
                out.append([val1, val2])

                listpt = merge_listpoints(listpt, pt1, pt2, lind1, lind2)
                # Merged lines, so don't check the other pairs
                break
            else:
                continue

    return lines, np.array(listpt), np.array(out)



