import cv2 as cv2
import numpy as np
import util as util
import curv_disc as cd
import depth_disc as dd


def create_img(mat):
    blank_image = np.zeros((mat.shape[0], mat.shape[1], 3), np.uint8)
    # print(blank_image.shape)
    mask = np.array(mat * 255, dtype=np.uint8)
    masked = np.ma.masked_where(mask <= 0, mask)

    return mask
    

def edge_detect(depth):
    curve_disc, curve_con = cd.curve_discont(depth)
    depth_disc, depth_con = dd.depth_discont(depth)

    # squeeze_ndarr(curve_con)
    # squeeze_ndarr(depth_con)

    # combine both images
    dst = (np.logical_or(curve_disc, depth_disc)).astype('uint8')
    dst = create_img(dst)
    # showimg(dst, "Depth + Discontinuity")
    skel_dst = util.morpho(dst)
    out = mask_contours(create_img(skel_dst))
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
    # squeeze_ndarr(res)

    dst = util.find_contours(create_img(skel_dst), cv2.RETR_EXTERNAL)
    # util.showimg(dst, "Depth + Discontinuity")


    return curve_disc, curve_con, depth_disc, depth_con, res


def mask_contours(im):
    # showimg(im)
    height = im.shape[0]
    width = im.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntrs = []
    cntrs.append(contours)
    cntr1 = contours
    cv2.drawContours(im, contours, -1, (0, 0, 0), 1, 8)
    cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 1, 8)
    # cv2.imshow("CONTOURS", blank_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    blank_image = np.zeros((height, width, 3), np.uint8)
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(hierarchy)
    # print(contours)
    # draw_contours(blank_image, contours)
    cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 1, 8)
    # cv2.imshow("CONTOURS 2", blank_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cntrs.append(contours)
    cntr2 = contours

    return cntr1 + cntr2