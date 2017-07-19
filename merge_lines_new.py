import numpy as np
from math import sqrt, atan, degrees, inf
from collections import Counter
from itertools import combinations, chain


def compare(s, t):
    return Counter(s) == Counter(t)


def math_stuff(x1, y1, x2, y2):
    slope = float((y2 - y1) / (x2 - x1) if ((x2 - x1) != 0) else inf)
    line_len = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = degrees(atan(-slope))
    return slope, line_len, alpha


def relevant_lines(pair, lines):
    line1 = pair[0]
    line2 = pair[1]
    data1 = lines[line1]
    data2 = lines[line2]
    alph1 = data1[6]
    alph2 = data2[6]
    lind1 = [data1[8], data1[9]]
    lind2 = [data2[8], data2[9]]
    return line1, line2, alph1, alph2, lind1, lind2


def merge_lines(list_lines, list_points, thres_angle, im_size):
    can_merge = True
    # Store the index of lines that get merged
    merged = [[n] for n in range(0, list_lines.shape[0])]
    while can_merge:
        # Stops if nothing gets merged
        can_merge = False

        unique_pts = np.sort(np.unique(list_lines[:, 8:10]))

        for i, lp in enumerate(unique_pts):
            # Test each combination of lines containing
            # this point to see which can be merged.
            # This is combinations w/o repetitions (choose 2)
            pairs = list(combinations(list(np.where(list_lines == lp)[0]), 2))
            # Check next point if no pairs were found for this point.
            if not pairs:
                continue
            for i, curr_pair in enumerate(pairs):
                line1, line2, alph1, alph2, lind1, lind2 = relevant_lines(curr_pair, list_lines)
                # Check that the lines are within the angle threshold and not coincident
                if abs(alph1 - alph2) > thres_angle or compare(lind1, lind2):
                    continue

                # This gets one start and one end point for the new line
                lind11, lind22 = np.sort([int(i) for i in list(filter(lambda e: e not in [lp], chain(lind1 + lind2)))])

                y1, x1 = np.unravel_index([lind1], im_size, order='F')
                y2, x2 = np.unravel_index([lind2], im_size, order='F')
                # print('y1', y1, 'x1', x1, 'y2', y2, 'x2', x2)
                # print(unravel_index([lind1], imgsize, order='F'))
                slope, line_len, alpha = math_stuff(x1, y1, x2, y2)

                # Intersection point is in the middle of the new line
                if min(alph1, alph2) <= alpha <= max(alph1, alph2):
                    list_lines = np.delete(list_lines, max(line1, line2), axis=0)
                    list_lines = np.delete(list_lines, min(line1, line2), axis=0)

                    val1 = merged[line1]
                    val2 = merged[line2]
                    del merged[max(line1, line2)]
                    del merged[min(line1, line2)]

                    # Update both lists to reflect the addition of the merged line.
                    list_lines = np.append(list_lines, [
                        [int(y1), int(x1) + 1, int(y2), int(x2) + 1, line_len, slope, alpha, 0, lind1, lind2]], axis=0)
                    merged.append([val1, val2])
                    can_merge = True
                    break
                else:
                    continue
