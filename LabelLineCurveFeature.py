
import numpy as np
import scipy.io as sio

# Written 11/8/2016
# Label Line curve feature
# Pre-condition: Gets the depth image and list of line segments and parameters
# Post-condition: Returns a list of lines labeled as curvature edges or discontinuities
# ----------------
# We define curvature edges as lines (or curves) on the object and discontinuities as those lines
# on the outside edges of the object where you can see it touching the background.

# 12/5/2016 - Gets some of the lines, but not all and doesn't get any lines past a particular index


def roipoly(src, line, poly):
    mask = []
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])
    dyy = line[0] - line[2]
    dxx = line[1] - line[3]

    if dy > dx or dy == dx:
        xfp = min(int(poly[0][1]), int(poly[1][1])) if dxx * dyy > 0 else max(int(poly[0][1]), int(poly[1][1]))
        mask_len = int(poly[3][1] - poly[0][1])
        y_range_start = min(int(poly[0][0]), int(poly[1][0]))
        y_range_end = max(int(poly[0][0]), int(poly[1][0]))

        for i in range(y_range_start, y_range_end):
            x0 = int(round(xfp))
            mask += list(src[i, x0:x0 + mask_len])
            step = (poly[1][1] - poly[0][1] + 0.0) / (poly[1][0] - poly[0][0] + 0.0)
            xfp += step
    else:
        yfp = min(int(poly[0][0]), int(poly[1][0])) if dxx * dyy > 0 else max(int(poly[0][0]), int(poly[1][0]))
        mask_len = int(poly[3][0] - poly[0][0])
        x_range_start = min(int(poly[0][1]), int(poly[1][1]))
        x_range_end = max(int(poly[0][1]), int(poly[1][1]))

        for i in range(x_range_start, x_range_end):
            y0 = int(round(yfp))
            mask += list(src[y0:y0 + mask_len, i])
            step = (poly[1][0] - poly[0][0] + 0.0) / (poly[1][1] - poly[0][1] + 0.0)
            yfp += step
    return mask

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
    return np.array([pt1, pt3, pt4, pt2]) if temp1 > temp2 else np.array([pt1, pt4, pt3, pt2])

def check_output(mat_out, py_out, check_dim):
    m_data = sio.loadmat(mat_out)
    m_dim = m_data[check_dim]

    p_out = np.where(np.asanyarray(py_out)[:, 10] == 12)
    m_out = np.where(np.asanyarray(m_dim)[:, 10] == 12)

    print('python lines:', p_out)
    print('matlab lines:', m_out)
    misses = 0
    extline = []
    lostline = []
    for i in py_out[0]:
        if (i in m_out[0]) == False:
            extline += [i]
    for i in m_out[0]:
        if (i in py_out[0]) == False:
            lostline += [i]

    print('extline:', len(extline))
    print('lostline:', len(lostline))

def classify_curves(src, list_lines, list_points, window_size):
    im_size = src.shape
    out = []
    for index, line in enumerate(list_lines):
        pt1, pt2, pt3, pt4 = get_orientation(line, window_size)
        win = get_ordering(pt1, pt2, pt3, pt4)

        mask4 = roipoly(src, line, win)
        mask4[:] = (value for value in mask4 if value != 0)

        a1 = np.mean(mask4)
        lx = list_points[index]
        temp_list = []
        for ii in lx[0]:
            temp3 = np.unravel_index(ii, im_size, order='F')
            temp4 = (temp3[0], temp3[1])
            temp_list.append(temp4)
        mask5 = []

        for i in temp_list:
            mask5.append(src[i])

        mask5[:] = (value for value in mask5 if value != 0)
        a2 = np.mean(mask5)
        b1 = len(mask4) * a1 - len(mask5) * a2
        try:
            b11 = float(b1) / (len(mask4) - len(mask5))
        except ZeroDivisionError:
            b11 = float('nan')

        if b11 < a2:
            out.append(np.append(list_lines[index], [12]))
        else:
            out.append(np.append(list_lines[index], [13]))

    check_output('linenewout.mat', np.asarray(out), 'Line_newC')
    return np.asarray(out)
