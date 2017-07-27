import cv2
import numpy as np
import util as util


if __name__ == '__main__':
    im = cv2.imread('img/learn1.png', -1)
    mask = np.ma.masked_less_equal(im, 0)
    print(mask.shape, "mask shape")
    print(mask.nonzero(), "non zero")
    # mask = np.array(mask.nonzero() * 255, dtype=np.uint8)
    print(im[48, 0], "point")

    white_bg = np.ones((im.shape[0], im.shape[1], 3), np.uint8) * 255
    # new_im = white_bg[mask]
    cv2.imshow("mask", im)
    cv2.waitKey()
    cv2.destroyAllWindows()