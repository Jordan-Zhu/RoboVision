from utility import *
from lineseg import lineseg
from drawedgelist import drawedgelist


# def draw_contours(im, contours):
#     height = im.shape[0]
#     width = im.shape[1]
#     blank_image = np.zeros((height, width, 3), np.uint8)
#     cimg = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#
#     for i in range(len(contours)):
#         # color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
#         cv2.drawContours(blank_image, contours, i, (255, 255, 255), 1, 8)
#
#
#     cv2.imshow("contours", dst)
#     cv2.waitKey(0)


if __name__ == '__main__':
    im1 = cv2.imread('lambda.png')
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray1, (3, 3), 0)

    can1 = auto_canny(blurred1, sigma=1)
    can1 = create_img(morpho(can1))

    cv2.imshow("Slide1", im1)
    cv2.imshow("Edges 1", can1)
    cv2.waitKey(0)

    height = im1.shape[0]
    width = im1.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)

    # ctr1 = mask_contours(can1)
    im2, ctr1, hierarchy = cv2.findContours(can1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res = []
    for i in range(len(ctr1)):
        current = np.squeeze(ctr1[i])
        if current.shape[0] > 2:
            res.append(current)
    res = np.array(res)
    print('before', res)
    swap_cols(res[0], 0, 1)
    print('after', res)
    print('contour 1: ', res.shape)
    print('hierarchy', hierarchy)

    res = lineseg(res, tol=2)
    seglist = []
    for i in range(res.shape[0]):
        if res[i].shape[0] > 2:
            seglist.append(np.concatenate((res[i], [res[i][0]])))
        else:
            seglist.append(res[i])
    seglist = np.squeeze(np.array(seglist))
    # swap_cols(seglist, 0, 1)
    seglist = np.array([seglist])
    # print(seglist)
    # print('after', seglist[0])
    # dst = np.logical_or(res, can1)
    draw_contours(can1, res)
    drawedgelist(seglist)

    # print('seglist:', res.shape)