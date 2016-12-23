import cv2
import numpy as np
import scipy.io as sio

from utility import roipoly, get_orientation, get_ordering

# Written 11/8/2016
# Label Line curve feature
# Pre-condition: Gets the depth image and list of line segments and parameters
# Post-condition: Returns a list of lines labeled as curvature edges or discontinuities
# ----------------
# We define curvature edges as lines (or curves) on the object and discontinuities as those lines
# on the outside edges of the object where you can see it touching the background.

# 12/5/2016 - Gets some of the lines, but not all and doesn't get any lines past a particular index


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

if __name__ == '__main__':
    depthimg = cv2.imread('img/learn15.png', -1)

    data = sio.loadmat('linenewin.mat')
    data2 = sio.loadmat('listpointin.mat')
    data3 = sio.loadmat('Id.mat')
    data4 = sio.loadmat('linenewout.mat')
    Line_new = list(data['Line_newC0'])
    ListPoint = data2['ListPoint_newC']
    Id = data3['Id']
    lout = data4['Line_newC']
    # Line_new, ListPoint, label_thresh

    line_new = classify_curves(depthimg, Line_new, ListPoint, 11)
