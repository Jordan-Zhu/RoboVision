import numpy as np
import cv2
from ZeroElimMedianFilter import zero_elim_median_filter
# from algo import showimg

def showimg(img, im_name='image', type='cv', write=False, imagename='img.png'):
    if type == 'cv':
        cv2.imshow(im_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if write:
            cv2.imwrite("../../images/%s", imagename, img)

def zeroElimMedianHoleFill(im):
    r = zero_elim_median_filter(im)
    has_holes = ~np.all(np.all(r))
    while has_holes:
        r = zero_elim_median_filter(r)
        has_holes = ~np.all(np.all(r))
    return r

if __name__ == '__main__':
    depthimg = cv2.imread('img/learn17.png', -1)
    h = zeroElimMedianHoleFill(depthimg)
    showimg(h)
