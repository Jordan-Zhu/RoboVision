from math import sqrt, atan, degrees
from numpy import sort, unique, where, unravel_index, delete, append
from itertools import combinations, chain
from collections import Counter


def compare(s, t):
    return Counter(s) == Counter(t)


def math_stuff(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    line_len = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    alpha = degrees(atan(-slope))
    return slope, line_len, alpha


def merge_listpoints(listpt, pt1, pt2):
    # Merge the list points
    lp1 = listpt[pt1]
    lp2 = listpt[pt2]
    startpt1 = list(where(lp1 == pt1)[0])
    startpt2 = list(where(lp1 == pt2)[0])
    startpt3 = list(where(lp2 == pt1)[0])
    startpt4 = list(where(lp2 == pt2)[0])

    if not startpt1:
        line_start = list(lp2)
        line_end = list(lp1)

        if startpt3[0] > 0:
            line_start = list(reversed(line_start))
        if startpt2[0] == 0:
            line_end = list(reversed(line_end))
    else:
        line_start = list(lp1)
        line_end = list(lp2)

        if startpt1[0] > 0:
            line_start = list(reversed(line_start))
        if startpt4[0] == 0:
            line_end = list(reversed(line_end))

    del listpt[max(pt1, pt2)]
    del listpt[min(pt1, pt2)]
    listpt.append(line_start[0:-1] + line_end)

    return listpt


def relevant_lines(i, pairs, lines):
    pt1 = pairs[i][0]
    pt2 = pairs[i][1]
    line1 = lines[pt1]
    line2 = lines[pt2]
    alph1 = line1[6]
    alph2 = line2[6]
    temp1 = [line1[8], line1[8]]
    temp2 = [line2[8], line2[8]]
    return pt1, pt2, alph1, alph2, temp1, temp2


def merge_lines(lines, listpt, thresh, imgsize):
    # lines format: y1, x1, y2, x2, length, slope, ???, index, start_pt, end_pt

    # All lines that can be merged. Merging lines
    # will group them together and then add the grouping to the list
    out = [[n] for n in range(0, lines.shape[0])]
    print(out)

    # Get unique start and end points. These are what we check
    unique_pts = sort(unique(lines[:, 8:10]))

    for index, ptx in enumerate(unique_pts):
        # Test each combination of lines with this
        # point to see which ones we can merge.
        # Formula is combinations w/o repetitions (choose 2)
        pairs = list(combinations(list(where(lines == ptx)[0]), 2))
        # Go to next iteration if there's no combinations
        if not pairs:
            continue
        for i in range(0, len(pairs)):
            pt1, pt2, alph1, alph2, temp1, temp2 = relevant_lines(i, pairs, lines)
            # Check that the lines are within the threshold and not coincident
            if abs(alph1 - alph2) > thresh or compare(temp1, temp2):
                continue

            px1, px2 = list(filter(lambda e: e not in [ptx], chain(temp1 + temp2)))
            y1, x1 = unravel_index([px1], imgsize, order='F')
            y2, x2 = unravel_index([px2], imgsize, order='F')
            slope, line_len, alpha = math_stuff(x1, y1, x2, y2)

            if min(alph1, alph2) <= alpha <= max(alph1, alph2):
                lines = delete(lines, [max(pt1, pt2), min(pt1, pt2)], axis=0)
                val1 = out[pt1]
                val2 = out[pt2]
                del out[min(pt1, pt2)]
                del out[max(pt1, pt2)]

                # Update both lists to reflect the addition of the merged line.
                lines = append(lines, [[y1, x1, y2, x2, line_len, slope, alpha, 0, px1, px2]], axis=0)
                out.append([val1, val2])

                listpt = merge_listpoints(listpt, pt1, pt2)











if __name__ == '__main__':
    lines_merged = [[n] for n in range(0, 100)]
    print(lines_merged)

