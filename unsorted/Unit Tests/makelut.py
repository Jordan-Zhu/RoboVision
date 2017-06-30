import numpy as np
import scipy.ndimage as ndimage
import cv2


def func(x):
    return (x==255).all()*255


if __name__ == '__main__':
    img = cv2.imread('neighborhood_example_1.png', 0)
    arr = ndimage.generic_filter(img, func, size=(2, 2)).astype('uint8')

    height = img.shape[0]
    width = img.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)

    cv2.imshow("CONTOURS", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()