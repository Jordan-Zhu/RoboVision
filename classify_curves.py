import cv2
import numpy as np
import util as util
import drawedgelist as de
import edge_detect as ed


def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)
    win = util.swap_indices(poly)
    cv2.fillConvexPoly(mask, win, 255)  # Create the ROI
    res = src * mask
    # cv2.imshow("roi", res)
    # cv2.waitKey(0)
    return res


def get_orientation(line, window_size):
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])
    # Vertical or horizontal line test
    if dy > dx or dy == dx:
        pt1 = [line[0], line[1] - window_size]
        pt2 = [line[0], line[1] + window_size]
        pt3 = [line[2], line[3] - window_size]
        pt4 = [line[2], line[3] + window_size]
        return pt1, pt2, pt3, pt4
    else:
        pt1 = [line[0] - window_size, line[1]]
        pt2 = [line[0] + window_size, line[1]]
        pt3 = [line[2] - window_size, line[3]]
        pt4 = [line[2] + window_size, line[3]]
        return pt1, pt2, pt3, pt4


def get_ordering(pt1, pt2, pt3, pt4):
    temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
    temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
    res = np.array([pt1, pt3, pt4, pt2]) if temp1 > temp2 else np.array([pt1, pt4, pt3, pt2])
    return [[int(i) for i in pt] for pt in res]


def get_lin_index(x1, y1, imgsize):
    return np.ravel_multi_index((y1, x1), imgsize, order='F')


# ref: https://stackoverflow.com/questions/29519726/find-matching-points-in-2-separate-numpy-arrays
def set_intersection(pA, pB):
    # Form concatenate array of pA and pB
    pts = np.concatenate((pA, pB), axis=0)

    # Sort pts by rows
    spts = pts[pts[:, 1].argsort(),]

    # Finally get counts by DIFFing along rows and counting all zero rows
    counts = np.sum(np.diff(np.all(np.diff(spts, axis=0) == 0, 1) + 0) == 1)

    return counts


def count_roi(im, win):
    roi = roipoly(im, win)
    line_c = np.where(roi == 255)
    pts = np.array(get_lin_index(line_c[1], line_c[0], roi.shape))
    return len(pts)


def classify_curves(curve_im, depth_im, list_lines, list_points, window_size):
    line_new = []
    # show both images so the user can see them
    cv2.imshow("curve im", ed.create_img(curve_im))
    cv2.imshow("depth im", ed.create_img(depth_im))
    # going through each line
    for i, line in enumerate(list_lines):
        # strategy:
        # check the curve and depth image
        # whichever one has more tells us what kind of line it is

        pt1, pt2, pt3, pt4 = get_orientation(line, window_size)
        win = np.array(get_ordering(pt1, pt2, pt3, pt4))

        count_c = count_roi(curve_im, win)
        count_d = count_roi(depth_im, win)

        # 5 is just the number that captured most of the holes
        if abs(count_c - count_d) <= 5:
            line_new.append(np.append(list_lines[i], [14]))
            # print("14, hole", count_c, count_d)
        elif count_c > count_d:
            # Line is a curvature
            line_new.append(np.append(list_lines[i], [12]))
            # print("12, curv", count_c, count_d)
        else:
            # discontinuity
            line_new.append(np.append(list_lines[i], [13]))
            # print("13, disc", count_c, count_d)

    return np.array(line_new)




