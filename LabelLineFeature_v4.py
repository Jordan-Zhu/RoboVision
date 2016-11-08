import numpy as np
import cv2
import scipy.io as sio
import math
# import roipoly

data = sio.loadmat('lableline_in_DE1.mat')  # canny edge detect on depth image
data2 = sio.loadmat('lableline_in_Id.mat')  # depth image
data3 = sio.loadmat('lableline_in_Line_new.mat')
data4 = sio.loadmat('lableline_out_Line_new.mat')
DE1 = data['DE1']
Id = data2['Id']
l2 = list(data3['Line_new'])
lout = data4['Line_new']
siz = (424 , 512)

Cons_Lmin = 12
label_win_sized = 3
thresh_label_dis = 0.08


def roipolyy(vx,Img):
    mask4 = []
    if (dy > dx) or (dy == dx):
        if dxx * dyy > 0:
            xfp = min(int(vx[0][1]), int(vx[1][1]))  ## original
        else:
            xfp = max(int(vx[0][1]), int(vx[1][1]))  ## original
        lenn = int(vx[3][1] - vx[0][1])
        yrangestart = min(int(vx[0][0]), int(vx[1][0]))
        yrangeend = max(int(vx[0][0]), int(vx[1][0]))

        for i in range(yrangestart, yrangeend):  # interation for dy
            x0 = int(round(xfp))
            mask4 += list(Img[i, x0:x0 + lenn])
            step = (vx[1][1] - vx[0][1] + 0.0) / (vx[1][0] - vx[0][0] + 0.0)
            xfp += step
            # len_mask = (yrangeend - yrangestart) * 2 * label_win_sized
    else:
        if dxx * dyy > 0:
            yfp = min(int(vx[0][0]), int(vx[1][0]))

        else:
            yfp = max(int(vx[0][0]), int(vx[1][0]))

        lenn = int(vx[3][0] - vx[0][0])
        xrangestart = min(int(vx[0][1]), int(vx[1][1]))
        xrangeend = max(int(vx[0][1]), int(vx[1][1]))

        for i in range(xrangestart, xrangeend):  # interation for dx
            y0 = int(round(yfp))

            mask4 += list(Img[y0:y0 + lenn, i])
            step = (vx[1][0] - vx[0][0] + 0.0) / (vx[1][1] - vx[0][1] + 0.0)
            yfp += step
            # len_mask = (xrangeend-xrangestart)*2*label_win_sized

    # mask4[:] = (value for value in mask4 if value != 0)

    return mask4





l = []
for i in l2:
    l.append(list(i)+[0,0])

# for cc in range(len(l)):
for cc in [180]:
    ul = l[cc]
    if ul[4] > Cons_Lmin:
        dy = abs(ul[0] - ul[2])
        dyy = ul[0] - ul[2]
        dx = abs(ul[1] - ul[3])
        dxx = ul[1] - ul[3]
        p1 = [ul[0],ul[1]]
        p2 = [ul[2], ul[3]]

        if (dy > dx) or (dy == dx):
            ptd1 = [ul[0], ul[1] - label_win_sized]
            ptd2 = [ul[0], ul[1] + label_win_sized]
            ptd3 = [ul[2], ul[3] - label_win_sized]
            ptd4 = [ul[2], ul[3] + label_win_sized]
            l[cc][11] = 1
        else:
            ptd1 = [ul[0] - label_win_sized, ul[1]]
            ptd2 = [ul[0] + label_win_sized, ul[1]]
            ptd3 = [ul[2] - label_win_sized, ul[3]]
            ptd4 = [ul[2] + label_win_sized, ul[3]]
            l[cc][11] = 2

        temp1 = np.linalg.norm(np.subtract((np.add(ptd1, ptd3) / 2.0), (np.add(ptd2, ptd4) / 2.0)))
        temp2 = np.linalg.norm(np.subtract((np.add(ptd1, ptd4) / 2.0), (np.add(ptd2, ptd3) / 2.0)))
        if temp1 > temp2:
            vxd = [ptd1,ptd3,ptd4,ptd2]
            winp = [p1,p2,ptd4,ptd2]
            winn = [ptd1,ptd3,p2,p1]
        else:
            vxd = [ptd1,ptd4,ptd3,ptd2]
            winp = [p1,ptd4,p2,ptd2]
            winn = [ptd1,p2,ptd3,p1]



        mask0 = roipolyy(vxd, DE1)
        if (dy > dx) or (dy == dx):
            yrangestart = min(int(vxd[0][0]), int(vxd[1][0]))
            yrangeend = max(int(vxd[0][0]), int(vxd[1][0]))
            yrangestart = ptd1[0]
            yrangeend = ptd1[1]
            # len_mask = (yrangeend - yrangestart) * 2 * label_win_sized
            len_mask = abs(yrangestart - yrangeend) * 2 * label_win_sized
        else:
            xrangestart = min(int(vxd[0][1]), int(vxd[1][1]))
            xrangeend = max(int(vxd[0][1]), int(vxd[1][1]))
            # len_mask = (xrangeend-xrangestart)*2*label_win_sized
            len_mask = abs(xrangestart - xrangeend) * 2 * label_win_sized

        tdd = len(mask0)/float(len_mask)
        if tdd>thresh_label_dis:
            maskp = roipolyy(winp,Id)
            maskn = roipolyy(winn, Id)

            dp = sum(maskp) / len(maskp)
            # print(dp)
            dn = sum(maskn) / len(maskn)
            if dp > dn:
                l[cc][10] = 9
            elif dp < dn:
                l[cc][10] = 10
            # if (len(maskp)>0)&(len(maskn)>0):
            #     dp = sum(maskp)/len(maskp)
            #     dn = sum(maskn) / len(maskn)
            #     if dp>dn:
            #         l[cc][10] = 9
            #     elif dp<dn:
            #         l[cc][10] = 10
            # else:
            #     print("mask is 0")
        else:
            l[cc][10] =  13


ll = np.asarray(l)
col10 = ll[:,10]
col11 = ll[:,11]
col10out = lout[:,10]
col11out = lout[:,11]
extline = []
lostline = []
for i in range(len(ll)):
    print(i, ".", ll[i])
for i in range(len(col10)):
    if col10[i] != col10out[i]:
        extline += [i]
for i in range(len(col11)):
    if col11[i] != col11out[i]:
        lostline += [i]

print(len(extline))
print(len(lostline))


a = 1

