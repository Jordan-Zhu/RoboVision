import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import util as util
import edge_detect
import lineseg
import drawedgelist

# img = cv2.imread("img/Slide2.jpg", 0)
img = cv2.imread("unsorted/Unit Tests/lambda.png", 0)
im_size = img.shape
returnedCanny = cv2.Canny(img, 50, 150, apertureSize = 3)

cv2.imshow("newcanny", returnedCanny)

skel_dst = util.morpho(returnedCanny)
out = edge_detect.mask_contours(edge_detect.create_img(skel_dst))




res = []
# print(np.squeeze(out[0]))
# print(out[0][0])
for i in range(len(out)):
    # Add the first point to the end so the shape closes
    current = np.squeeze(out[i])
    # print('current', current)
    # print('first', out[i][0])
    if current.shape[0] > 2:
        # res.append(np.concatenate((current, out[i][0])))
        # print(res[-1])
        res.append(current)
    # print(np.concatenate((np.squeeze(out[i]), out[i][0])))

res = np.array(res)
util.sqz_contours(res)

res = lineseg.lineseg(np.array([res[1]]), tol=5)
print(res, "res")
"""
for x in range(len(res)):
    for y in range(lan ):
"""

drawedgelist.drawedgelist(res, img)




"""

seglist = []
for i in range(res.shape[0]):
    # print('shape', res[i].shape)
    if res[i].shape[0] > 2:
        # print(res[i])
        # print(res[i][0])
        seglist.append(np.concatenate((res[i], [res[i][0]])))
    else:
        seglist.append(res[i])

seglist = np.array(seglist)


"""
#print(seglist, "seglist")
#print(len(seglist), "seglist len")
#print(seglist.shape, "seglistshape")
#drawedgelist.drawedgelist(seglist)

"""
# ******* SECTION 2 *******
# SEGMENT AND LABEL THE CURVATURE LINES (CONVEX/CONCAVE).
LineFeature, ListPoint = Lseg_to_Lfeat_v4.create_linefeatures(seglist, res, im_size)
Line_new, ListPoint_new, line_merged = merge_lines_v4.merge_lines(LineFeature, ListPoint, 10, im_size)

#print(Line_new, "line new")
print(len(Line_new), "len line new")
util.draw_lf(Line_new, blank_image)

line_newC = LabelLineCurveFeature_v4.classify_curves(img, Line_new, ListPoint_new, 11)"""