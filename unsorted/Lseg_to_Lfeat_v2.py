import numpy as np
import cv2
import scipy.io as sio
import math
# from math_utils import sub2ind

# data = sio.loadmat('test.mat')
# data2 = sio.loadmat('test2.mat')
# l = list(data['ListSegLineC'])
# l2 = list(data2['ListEdgeC'])
# siz = (424,512)
#
# ll = []
# for i in l[0]:
#     ll.append(i)
# ll2 = []
# for i in l2[0]:
#     ll2.append(i)
#

import warnings
warnings.filterwarnings("ignore")

def Lseg_to_Lfeat_v2(ll,ll2,siz):
    c0 = 1
    Linefeature = []
    ListPoint = []
    # for cc in range(len(l[0])):#[0]
    for cc in range(len(ll)):
        temp = ll[cc]  #l[0]
        ax = len(temp)

        for c2 in range(ax-1):

            x1 = temp[c2,0].astype(int)
            x2 = temp[c2+1,0].astype(int)
            y1 = temp[c2,1].astype(int)
            y2 = temp[c2+1,1].astype(int)

            m = round((y2-y1)/float((x2-x1)),2)
            lind1 = np.ravel_multi_index((y1,x1),siz,order= 'F')          #  np.unravel_index()
            lind2 = np.ravel_multi_index((y2,x2),siz,order= 'F')
            L = round(math.sqrt(round((x2-x1)**2+(y2-y1)**2,2)),2)
            alpha = round(math.atan(-m)*180/math.pi,2)

            if c0>2:
                if ((lind1 != Linefeature[c0-2][8]) or (lind2 != Linefeature[c0-2][9])):
                    # print c0
                    Linefeature.append([y1,x1,y2,x2,L,m,alpha,c0,lind1,lind2])
                    c0 += 1
            else:
                Linefeature.append([y1, x1, y2, x2, L, m, alpha, c0, lind1, lind2])
                c0 += 1

            # sty = np.where(l2[0][cc][:,0]==y1)# [0]
            # stx = np.where(l2[0][cc][:,1] == x1)
            # a = set(sty[0]).intersection(stx[0])
            #
            # sty = np.where(l2[0][cc][:,0] == y2)
            # stx = np.where(l2[0][cc][:,1] == x2)
            # b = set(sty[0]).intersection(stx[0])
            sty = np.where(ll2[cc][:,0, 0] == x1)  # [0]
            stx = np.where(ll2[cc][:, 0,1] == y1)
            a = set(sty[0]).intersection(stx[0])

            sty = np.where(ll2[cc][:, 0,0] == x2)
            stx = np.where(ll2[cc][:, 0,1] == y2)
            b = set(sty[0]).intersection(stx[0])



            # ListPoint.append([l2[0][cc]])
            ListPoint.append([ll2[cc]])


            # if c0>2:
            #     if ((lind1==Linefeature[c0-2][8]) and (lind2==Linefeature[c0-2][9])):
            #         c0 = c0 - 1
                    # print 'Y'

    lx = len(ListPoint)
    LPP = []
    for cnt in range(lx):
        LPP.append([np.ravel_multi_index((ListPoint[cnt][0][:, 0, 1], ListPoint[cnt][0][:,0,  0]), siz,order= 'F')])
    return Linefeature,LPP

# [Line,Lp] = Lseg_to_Lfeat_v2(ll,ll2,siz)


# c0 = 1
# Linefeature = []
# ListPoint = []
# for cc in range(len(l)):  # [0]
#     temp = l[cc]  # l[0]
#     ax = len(temp)
#
#     # print cc
#     for c2 in range(ax - 1):
#         y1 = temp[c2, 0].astype(int)
#         y2 = temp[c2 + 1, 0].astype(int)
#         x1 = temp[c2, 1].astype(int)
#         x2 = temp[c2 + 1, 1].astype(int)
#         m = round((y2 - y1) / float((x2 - x1)), 2)
#         # print cc
#         # print c2
#         lind1 = np.ravel_multi_index((y1, x1), siz[::-1])  # np.unravel_index()
#         lind2 = np.ravel_multi_index((y2, x2), siz[::-1])
#         L = round(math.sqrt(round((x2 - x1) ** 2 + (y2 - y1) ** 2, 2)), 2)
#         alpha = round(math.atan(-m) * 180 / math.pi, 2)
#
#         if c0 > 2:
#             if ((lind1 != Linefeature[c0 - 2][8]) or (lind2 != Linefeature[c0 - 2][9])):
#                 # print c0
#                 Linefeature.append([y1, x1, y2, x2, L, m, alpha, c0, lind1, lind2])
#                 c0 += 1
#         else:
#             Linefeature.append([y1, x1, y2, x2, L, m, alpha, c0, lind1, lind2])
#             c0 += 1
#
#         sty = np.where(l2[cc][:, 0,0] == y1)  # [0]
#         stx = np.where(l2[cc][:, 0,1] == x1)
#         a = set(sty[0]).intersection(stx[0])
#
#         sty = np.where(l2[cc][:, 0,0] == y2)
#         stx = np.where(l2[cc][:, 0,1] == x2)
#         b = set(sty[0]).intersection(stx[0])
#
#         ListPoint.append([l2[cc]])
#
#
#
#         # if c0>2:
#         #     if ((lind1==Linefeature[c0-2][8]) and (lind2==Linefeature[c0-2][9])):
#         #         c0 = c0 - 1
#         # print 'Y'
#
# lx = len(ListPoint)
# LPP = []
# for cnt in range(lx):
#     LPP.append([np.ravel_multi_index((ListPoint[cnt][0][:, 0,1], ListPoint[cnt][0][:, 0,0]), siz)])