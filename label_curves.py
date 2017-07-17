import cv2
import numpy as np
import util as util

global window_size


def vertical_line(line):
    line[11] = 1
    # [y ; x-ts]
    return [line[0], line[1] - window_size], \
           [line[0], line[1] + window_size], \
           [line[2], line[3] - window_size], \
           [line[2], line[3] + window_size]


def horizontal_line(line):
    line[11] = 2
    # [y-ts ; x]
    return [line[0] - window_size, line[1]], \
           [line[0] + window_size, line[1]], \
           [line[2] - window_size, line[3]], \
           [line[2] + window_size, line[3]]

def get_orientation(line):
    startpt = [line[0], line[1]]
    endpt = [line[2], line[3]]
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])

    if dy > dx or dy == dx:
        pt1, pt2, pt3, pt4 = vertical_line(line)
    else:
        pt1, pt2, pt3, pt4 = horizontal_line(line)
    return pt1, pt2, pt3, pt4, startpt, endpt


def create_windows(pt1, pt2, pt3, pt4, startpt, endpt):
    temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
    temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
    if temp1 > temp2:
        win_p = [startpt, endpt, pt4, pt2]
        win_n = [pt1, pt3, endpt, startpt]
    else:
        win_p = [startpt, pt4, endpt, pt2]
        win_n = [pt1, endpt, pt3, startpt]
    return win_p, win_n


def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)
    win = util.swap_indices(poly).astype(int)

    cv2.fillConvexPoly(mask, win, 255)  # Create the ROI
    return mask


def mask_mean(src, mask):
    val_mask = src * mask
    if cv2.countNonZero(val_mask) == 0:
        return 0
    else:
        return sum(src[np.nonzero(val_mask)]) / cv2.countNonZero(val_mask)


def create_mask(src, win_p, win_n):
    mask_p = roipoly(src, win_p)
    mask_n = roipoly(src, win_n)
    mean_p = mask_mean(src, mask_p)
    mean_n = mask_mean(src, mask_n)
    return mean_p, mean_n

# concave/convex of a curvature
def label_convexity(lp_curr, mask_p, mask_n):
    mean_win = (mask_p + mask_n) / 2
    # if lp_curr > mean_win:
    #     # convex
    #     return 3
    # elif lp_curr < mean_win:
    #     # concave
    #     return 4
    # else:
    #     print("wat")
    #     return -1
    return 3 if lp_curr >= mean_win else 4


# obj on left/right side of discontinuity
def label_pose(mask_p, mask_n):
    return 1 if mask_p >= mask_n else 2



def label_curves(src, list_lines, list_point):
    global window_size
    window_size = 5
    # strategy:
    # make window on both sides of line
    # run test for curv or disc
    # append results of test to col 12
    col_label = np.zeros((list_lines.shape[0], 2))
    list_lines = np.hstack((list_lines, col_label))
    # print(list_lines, "Lines")
    for i, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4, startpt, endpt = get_orientation(line)
        win_p, win_n = create_windows(pt1, pt2, pt3, pt4, startpt, endpt)
        # print(win_p, "win p", win_n, "win n")
        mean_p, mean_n = create_mask(src, win_p, win_n)
        # print(mean_p, 'mean p', mean_n, 'mean n')

        if line[10] == 12:
            y, x = np.unravel_index([list_point[i]], src.shape, order='F')
            mean_lp = np.mean(src[y, x])
            # print(mean_lp, "mean lp")
            # mean_lp = np.mean(src[list_point[i]])
            label = label_convexity(mean_lp, mean_p, mean_n)
        elif line[10] == 13:
            label = label_pose(mean_p, mean_n)
        else:
            label = 0
        line[12] = label

    return list_lines