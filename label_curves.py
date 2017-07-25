import cv2
import numpy as np
import util as util

global window_size
global buffer_zone


def vertical_line(line):
    line[11] = 1
    # [y ; x-ts]
    return [line[0] - buffer_zone, line[1] - window_size - buffer_zone], \
           [line[0] + buffer_zone, line[1] + window_size + buffer_zone], \
           [line[2] - buffer_zone, line[3] - window_size - buffer_zone], \
           [line[2] + buffer_zone, line[3] + window_size + buffer_zone]


def horizontal_line(line):
    line[11] = 2
    # [y-ts ; x]
    return [line[0] - window_size - buffer_zone, line[1] - buffer_zone], \
           [line[0] + window_size + buffer_zone, line[1] + buffer_zone], \
           [line[2] - window_size - buffer_zone, line[3] - buffer_zone], \
           [line[2] + window_size + buffer_zone, line[3] + buffer_zone]

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
    mask_size = cv2.countNonZero(mask)
    num_nonzero = cv2.countNonZero(val_mask)
    if num_nonzero == 0 or num_nonzero / mask_size < 0.05:
        return 100000
    else:
        return sum(src[np.nonzero(val_mask)]) / num_nonzero


def remove_points(lp, roi):
    line_mask = np.invert(np.in1d(lp, roi))
    # print(lp, "lp")
    # print(line_mask, "line mask")
    roi = roi[:][line_mask]
    cv2.imshow("roi after", roi)

def create_mask(src, lp, win_p, win_n):
    mask_p = roipoly(src, win_p)
    mask_n = roipoly(src, win_n)
    # remove_points(lp, mask_p)
    mean_p = mask_mean(src, mask_p)
    mean_n = mask_mean(src, mask_n)
    return mean_p, mean_n

# concave/convex of a curvature
def label_convexity(lp_curr, mask_p, mask_n, points_p, points_n):
    # mean_win = (mask_p + mask_n) / 2
    thres = 0.99
    print("| LP mean:", lp_curr, "| mask_P:", mask_p, "| mask_N:", mask_n, "p count", points_p, "n count", points_n)
    if lp_curr <= mask_p and lp_curr <= mask_n:
        if mask_p >= mask_n and points_p >= points_n:
            return 31
        elif mask_n >= mask_p and points_n >= points_p:
            return 32
        else:
            return -1
        # return 31 if mask_p >= mask_n and points_p >= points_ else 32
    elif lp_curr > mask_p or lp_curr > mask_n:
        return 4
    # if lp_curr > mask_p and lp_curr > mask_n or lp_curr / mask_p > thres or lp_curr / mask_n > thres:
    #     return 4
    # elif lp_curr <= mask_p and lp_curr <= mask_n:
    #     return 31 if mask_p >= mask_n else 32
    # else:
    #     return 4
    # if lp_curr > mean_win:
    #     # convex
    #     return 3
    # elif lp_curr < mean_win:
    #     # concave
    #     return 4
    # else:
    #     print("wat")
    #     return -1
    # return 3 if lp_curr >= mean_win else 4


# obj on left/right side of discontinuity
def label_pose(mask_p, mask_n, points_p, points_n):
    if mask_p >= mask_n and points_p >= points_n:
        return 1
    elif mask_n <= mask_p and points_n <= points_p:
        return 2
    else:
        return -1
    # return 1 if mask_p >= mask_n else 2


# This removes boundary lines that aren't part of the shape
def remove_lines(src, contour, win_p, win_n):
    mask_p = roipoly(src, win_p)
    mask_n = roipoly(src, win_n)
    # cv2.imshow("mask_p", mask_p)
    # cv2.imshow("mask_n", mask_n)
    tp = np.nonzero(mask_p)
    tn = np.nonzero(mask_n)
    print(np.nonzero(mask_p), "win p")
    print(np.nonzero(mask_n), "win n")
    pixels_p = np.squeeze(np.dstack((tp[0], tp[1])))
    pixels_n = np.squeeze(np.dstack((tn[0], tn[1])))
    print(pixels_p, "pixels p")
    print(pixels_n, "pixels n")
    # mask_p = src * mask_p
    # mask_n = src * mask_n
    mask = np.zeros(src.shape, np.uint8)
    # print(contour, "contour")
    cv2.drawContours(mask, [contour], 0, 255, -1)
    cv2.imshow("mask", np.transpose(mask))
    pixelpoints = np.transpose(np.nonzero(mask))

    tc = np.nonzero(mask)
    contour = np.squeeze(np.dstack((tc[0], tc[1])))

    # points_p = np.count_nonzero(np.nonzero(mask_p) == np.nonzero(mask))
    # points_n = np.count_nonzero(np.nonzero(mask_n) == np.nonzero(mask))

    points_p = 0
    points_n = 0
    for z in range(len(pixels_p)):
        if (contour == pixels_p[z]).all(1).any():
            points_p += 1

    for z in range(len(pixels_n)):
        if (contour == pixels_n[z]).all(1).any():
            points_n += 1

    print("counts:\np = ", points_p, "\nn = ", points_n)


    # print(np.count_nonzero(np.nonzero(points_p)[0] == np.nonzero(mask)[0]), "nonzero")
    # print((np.nonzero(points_p)[0] == np.nonzero(mask)[0]).sum(), "nonzero")
    return points_p, points_n



def label_curves(src, list_lines, list_point, contour):
    global window_size
    window_size = 5
    global buffer_zone
    buffer_zone = 0
    # append results of test to col 12
    col_label = np.zeros((list_lines.shape[0], 2))
    list_lines = np.hstack((list_lines, col_label))
    # print(list_lines, "Lines")
    # print(contour, "contour")
    for i, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4, startpt, endpt = get_orientation(line)
        win_p, win_n = create_windows(pt1, pt2, pt3, pt4, startpt, endpt)
        # print(win_p, "win p", win_n, "win n")
        # y, x = np.unravel_index([list_point[i]], src.shape, order='F')
        # print(y, "y", x, "x")
        # print(list_point[i].shape)
        # pts = []
        # if len(list_point[i]) > 2:
        #     for i in range(len(y)):
        #         pts.append([y[0][i], x[0][i]])
        # elif len(list_point[i]) == 1:
        #     pts.append([y[0], x[0]])
        # print(pts)
        points_p, points_n = remove_lines(src, contour, win_p, win_n)
        mean_p, mean_n = create_mask(src, list_point[i], win_p, win_n)
        # print(mean_p, 'mean p', mean_n, 'mean n')

        if line[10] == 12:
            #print(len(list_point[i]), "num list point")
            y, x = np.unravel_index([list_point[i]], src.shape, order='F')
            # print("Y:", y, "\nX:", x)
            mean_lp = np.mean(src[y, x])
            # print(mean_lp, "mean lp")
            # mean_lp = np.mean(src[list_point[i]])
            label = label_convexity(mean_lp, mean_p, mean_n, points_p, points_n)
            print(label, "curv label")
        elif line[10] == 13 or line[10] == 14:
            label = label_pose(mean_p, mean_n, points_p, points_n)
        else:
            label = 0
        line[12] = label

    return list_lines
