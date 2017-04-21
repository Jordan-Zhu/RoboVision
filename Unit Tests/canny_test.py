from utility import *
from lineseg import lineseg
from drawedgelist import drawedgelist


if __name__ == '__main__':
    im1 = cv2.imread('Slide1.PNG')
    im2 = cv2.imread('Slide2.PNG')
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray1, (3, 3), 0)

    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    blurred2 = cv2.GaussianBlur(gray2, (3, 3), 0)

    can1 = auto_canny(blurred1, sigma=1)
    can2 = auto_canny(blurred2, sigma=1)

    cv2.imshow("Slide1", im1)
    cv2.imshow("Edges 1", can1)
    cv2.imshow("Slide2", im2)
    cv2.imshow("Edges 2", can2)
    cv2.waitKey(0)

    height = im1.shape[0]
    width = im1.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)

    ctr1 = mask_contours(create_img(morpho(can1)))
    res = []
    for i in range(len(ctr1)):
        current = np.squeeze(ctr1[i])
        if current.shape[0] > 2:
            res.append(current)
    res = np.array(res)

    res = lineseg(res, tol=2)
    seglist = []
    for i in range(res.shape[0]):
        if res[i].shape[0] > 2:
            seglist.append(np.concatenate((res[i], [res[i][0]])))
        else:
            seglist.append(res[i])
    seglist = np.array(seglist)
    drawedgelist(seglist)

    ctr2 = mask_contours(create_img(morpho(can2)))

    print(res.shape)
    draw_contours(blank_image, res)

    blank_image = np.zeros((height, width, 3), np.uint8)
    draw_contours(blank_image, ctr2)