from utility import *
from lineseg import lineseg
from drawedgelist import drawedgelist
from collections import OrderedDict
from itertools import groupby


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
    im1 = cv2.imread('bw_lambda.png')
    # gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # blurred1 = cv2.GaussianBlur(gray1, (3, 3), 0)

    # can1 = auto_canny(blurred1, sigma=1)
    # can1 = create_img(im1)

    # cv2.imshow("Slide1", im1)
    # cv2.imshow("Edges 1", can1)
    # cv2.waitKey(0)

    height = im1.shape[0]
    width = im1.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)

    # ctr1 = mask_contours(can1)
    im2, ctr1, hierarchy = cv2.findContours(im1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res = []
    for i in range(len(ctr1)):
        current = np.squeeze(ctr1[i])
        if current.shape[0] > 2:
            res.append(current)
    a = np.array(res)[0]

    print('res before', a)
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_a = a[idx]
    # print('unique', unique_a)
    x = a[:, 0]
    y = a[:, 1]
    # print('x', x)
    # print('y', y)

    tmp = OrderedDict()
    for point in zip(x, y):
        tmp.setdefault(point[:2], point)

    mypoints = tmp.values()

    # keyfunc = lambda p: p[:2]
    # mypoints = []
    # for k, g in groupby(sorted(zip(x, y), key=keyfunc), keyfunc):
    #     mypoints.append(list(g)[0])
    #
    arr = np.empty((len(tmp.keys()), 2))
    row_indices = tmp.keys()
    # arr = np.array(mypoints)
    for i, row in enumerate(row_indices):
        arr[i] = tmp[row]
    arr = arr.astype(int)
    print('arr', arr)


    for i in range(arr.shape[0] - 1):
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.line(blank_image, (arr[i][0], arr[i][1]), (arr[i][0], arr[i][1]), color, thickness=1)

    cv2.namedWindow("Edge list", cv2.WINDOW_NORMAL)
    cv2.imshow("Edge list", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    res = np.array([arr])
    print('res after shape', res.shape)
    # drawedgelist(res)
    # draw_contours(blank_image, res)
    # print('before', res)
    swap_cols(res[0], 0, 1)
    # print('after', res[0])
    # print('contour 1: ', res.shape)
    # print('hierarchy', hierarchy)

    res = lineseg(res, tol=2)
    # print('seglist', res)
    seglist = []
    for i in range(res.shape[0]):
        if res[i].shape[0] > 2:
            seglist.append(np.concatenate((res[i], [res[i][0]])))
        else:
            seglist.append(res[i])
    seglist = np.squeeze(np.array(seglist))
    # swap_cols(seglist, 0, 1)
    # seglist = np.array(seglist)
    print('seglist', seglist)

    x = seglist[:, 0]
    y = seglist[:, 1]
    # print('x', x)
    # print('y', y)

    tmp = OrderedDict()
    for point in zip(x, y):
        tmp.setdefault(point[:2], point)

    mypoints = tmp.values()

    keyfunc = lambda p: p[:2]
    mypoints = []
    for k, g in groupby(sorted(zip(x, y), key=keyfunc), keyfunc):
        mypoints.append(list(g)[0])

    arr = np.empty((len(tmp.keys()), 2))
    row_indices = tmp.keys()
    # arr = np.array(mypoints)
    for i, row in enumerate(row_indices):
        arr[i] = tmp[row]
    arr = np.array([arr.astype(int)])
    print('arr', arr)

    # print('after', seglist[0])
    drawedgelist([seglist])

    # print('seglist:', res.shape)