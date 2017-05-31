import numpy as np
import cv2
import scipy.io as sio
import math

# import roipoly
#
# data = sio.loadmat('linenewin.mat')
# data2 = sio.loadmat('listpointin.mat')
# data3 = sio.loadmat('Id.mat')
# data4 = sio.loadmat('linenewout.mat')
# Line_new = list(data['Line_newC0'])
# ListPoint = data2['ListPoint_newC']
# Id = data3['Id']
# lout = data4['Line_newC']
# siz = (424, 512)
# label_thresh = 11

def LabelLineCurveFeature_v2(Id,Line_new,ListPoint,label_thresh):
    siz = Id.shape
    Line_new = list(Line_new)
    for cc in range(len(Line_new)):
    # for cc in [0]:
        ul = Line_new[cc]
        dy = abs(ul[0] - ul[2])
        dyy = ul[0] - ul[2]
        dx = abs(ul[1] - ul[3])
        dxx = ul[1] - ul[3]
        if (dy > dx) or (dy == dx):
            pt1 = [ul[0], ul[1] - label_thresh]
            pt2 = [ul[0], ul[1] + label_thresh]
            pt3 = [ul[2], ul[3] - label_thresh]
            pt4 = [ul[2], ul[3] + label_thresh]
        else:
            pt1 = [ul[0] - label_thresh, ul[1]]
            pt2 = [ul[0] + label_thresh, ul[1]]
            pt3 = [ul[2] - label_thresh, ul[3]]
            pt4 = [ul[2] + label_thresh, ul[3]]

        temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
        temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
        if temp1 > temp2:
            vx = [pt1, pt3, pt4, pt2]
        else:
            vx = [pt1, pt4, pt3, pt2]

        # print vx
        # mask1 = 1
        # vx0 = np.asarray(vx)[:, 0]
        # vx1 = np.asarray(vx)[:, 1]
        vx = np.asarray(vx)
        # # print vx
        # xmin = vx[:,0].min()
        # xmax = vx[:, 0].max()
        # ymin = vx[:,1].min()
        # ymax = vx[:, 1].max()

        # mask4 = np.asarray(Ip)[vx[0][0].astype(int)-1:vx[2][0].astype(int)-1,vx[3][1].astype(int)-1:vx[1][1].astype(int)-1]
        # mask4 = Ip[xmin-1:xmax-1,ymin-1:ymax-1]      # need to be improved    now just rectangular

        ###  mask for polygon
        mask4 = []
        if (dy > dx) or (dy == dx):
            if dxx * dyy > 0:
                xfp = min(int(vx[0][1]), int(vx[1][1]))  ##original
                # xfp = min(int(vx[0][1]), int(vx[1][1]))-1
            else:
                xfp = max(int(vx[0][1]), int(vx[1][1]))  ## original
                # xfp = max(int(vx[0][1]), int(vx[1][1]))-1
            lenn = int(vx[3][1] - vx[0][1])
            yrangestart = min(int(vx[0][0]), int(vx[1][0]))
            yrangeend = max(int(vx[0][0]), int(vx[1][0]))
            # yrangestart = min(int(vx[0][0]), int(vx[1][0]))-1
            # yrangeend = max(int(vx[0][0]), int(vx[1][0]))-1

            for i in range(yrangestart, yrangeend):  # interation for dy
                x0 = int(round(xfp))
                # mask4 += list(Ip[i, x0 - 1:x0 - 1 + lenn])  ## original
                mask4 += list(Id[i, x0:x0 + lenn])
                # mask4.append(Ip[i, x0 - 1:x0 - 1+lenn])
                # if dxx*dyy>0:
                step = (vx[1][1] - vx[0][1] + 0.0) / (vx[1][0] - vx[0][0] + 0.0)
                # else:
                #     slop = -(vx[1][1] - vx[0][1] + 0.0) / (vx[1][0] - vx[0][0] + 0.0)
                xfp += step
                # print x0
        else:
            if dxx * dyy > 0:
                yfp = min(int(vx[0][0]), int(vx[1][0]))
                # yfp = min(int(vx[0][0]), int(vx[1][0]))-1

            else:
                yfp = max(int(vx[0][0]), int(vx[1][0]))
                # yfp = max(int(vx[0][0]), int(vx[1][0]))-1
            # yfp = min(int(vx[0][0]),int(vx[1][0]))
            lenn = int(vx[3][0] - vx[0][0])
            xrangestart = min(int(vx[0][1]), int(vx[1][1]))
            xrangeend = max(int(vx[0][1]), int(vx[1][1]))
            # xrangestart = min(int(vx[0][1]), int(vx[1][1]))-1
            # xrangeend = max(int(vx[0][1]), int(vx[1][1]))-1
            for j in range(xrangestart, xrangeend):  # interation for dx
                y0 = int(round(yfp))
                # mask4.append(Ip[y0 - 1:y0 - 1+lenn,i])
                # mask4 += list(Ip[y0 - 1:y0 - 1 + lenn, i])#  original
                mask4 += list(Id[y0:y0 + lenn, j])
                step = (vx[1][0] - vx[0][0] + 0.0) / (vx[1][1] - vx[0][1] + 0.0)
                yfp += step
                # print y0

        # mask44 = []
        # for i in mask4:
        #     # print i
        #     mask44 += list(i)
        # # print mask44
        # mask44[:] = (value for value in mask44 if value != 0)

        mask4[:] = (value for value in mask4 if value != 0)

        # print mask4
        A1 = np.mean(mask4)
        lind = ListPoint[cc]
        lind0 = []
        for ii in lind[0]:
            temp3 = np.unravel_index(ii, siz, order='F')
            # print temp3
            temp4 = (temp3[0] , temp3[1])  # +1 for mat file
            lind0.append(temp4)
        # np.asarray(Ip)[np.unravel_index(15066, siz)[::-1]]
        # print len(lind0)
        mask5 = []

        # print lind0
        for i2 in lind0:
            i1 = i2  # [::-1]
            mask5.append(Id[i1])

        mask5[:] = (value for value in mask5 if value != 0)
        # print mask5
        A2 = np.mean(mask5)
        B1 = len(mask4) * A1 - len(mask5) * A2
        try:
            B11 = float(B1) / (len(mask4) - len(mask5))
        except ZeroDivisionError:
            B11 = float('nan')
        # print A2
        # print B11<A2

        # if cc == 247:
        # print B11,A2
        if B11 < A2:  # Becasue the way to corp the region of interest is different
            Line_new[cc] = np.append(Line_new[cc], [12])  # so the B11 and A2 will slight different with the value in matlab
        else:  # when the value of B11 and A2 are very lose, the concave and convex feature may change
            Line_new[cc] = np.append(Line_new[cc], [13])  #

    return np.asarray(Line_new)
        # print B11
        # print A2
# ## compare
# out0 = np.where(np.asanyarray(Line_new)[:, 10] == 12)
# out1 = np.where(np.asanyarray(lout)[:, 10] == 12)
# extline = []
# lostline = []
# for i in out0[0]:
#     if (i in out1[0]) == False:
#         extline += [i]
# for i in out1[0]:
#     if (i in out0[0]) == False:
#         lostline += [i]
#
# print len(extline)
# print len(lostline)
#
# a = 1
