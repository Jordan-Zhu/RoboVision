import cv2
import numpy as np
import util as util
import edge_detect
import random
import lineseg
import drawedgelist
import Lseg_to_Lfeat_v4 as Lseg_to_Lfeat_v4


im = cv2.imread("test1.png", 0)
im_size = im.shape
returnedCanny = cv2.Canny(im, 50, 150, apertureSize=3)

cv2.imshow("newcanny", returnedCanny)

skel_dst = util.morpho(returnedCanny)
new_img = edge_detect.create_img(skel_dst)

im2, contours, hierarchy = cv2.findContours(new_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

height = im2.shape[0]
width = im2.shape[1]
blank_image = np.zeros((height, width, 3), np.uint8)
for x in range(len(contours)):
    randC = random.uniform(0, 1)
    randB = random.uniform(0, 1)
    randA = random.uniform(0, 1)
    cv2.drawContours(blank_image, contours, x, (int(randA * 255), int(randB * 255), int(randC * 255)), 1, 8)

cv2.imshow("CONTOURS", blank_image)
print(len(contours), "contours")
print(len(contours[0]), "contours")

out = contours

res = []
# print(np.squeeze(out[0]))
# print(out[0][0])
for i in range(len(out)):
    # Add the first point to the end so the shape closes
    current = np.squeeze(out[i])
    if current.shape[0] > 2:
        res.append(current)

res = np.array(res)
util.sqz_contours(res)

seglist = lineseg.lineseg(res, tol=2)
print(res, "res")
"""
for x in range(len(res)):
    for y in range(lan ):
"""

blank_image = np.zeros((height, width, 3), np.uint8)
drawedgelist.drawedgelist(seglist, blank_image)

# extract line features
LineFeature, ListPoint = Lseg_to_Lfeat_v4.create_linefeatures(seglist, res, im_size)
# print(LineFeature, "LineFeature")
print(LineFeature[0], "LineFeature[0]")
print(ListPoint[0], "ListPoint")
# print(res.shape, 'res shape')
# blank_image = np.zeros((height, width, 3), np.uint8)
# drawedgelist.drawedgelist([res[0]], blank_image)
# print(ListPoint.shape, "ListPoint size")
print(ListPoint[0][0], "lp start")
print(ListPoint[0][-1], "lp end")

print(LineFeature[0][0], LineFeature[0][1], "lf start")
print(LineFeature[0][2], LineFeature[0][3], "lf end")

y1, x1 = np.unravel_index([ListPoint[0][0]], im_size, order='F')
y2, x2 = np.unravel_index([ListPoint[0][-1]], im_size, order='F')

print(y1, x1, "starting point")
print(y2, x2, "ending point")

blank_image = np.zeros((height, width, 3), np.uint8)
util.draw_lf([LineFeature[0]], blank_image)


# cv2.waitKey(0)
# cv2.destroyAllWindows()