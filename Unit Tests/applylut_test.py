import numpy as np
import scipy.ndimage as ndimage
import cv2 as cv2
from cv2 import *


im = cv2.imread('e56Bt.png')
grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blank_image = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)

cv2.namedWindow('Before', cv2.WINDOW_NORMAL)
cv2.imshow('Before', im)
cv2.waitKey(0)

def func(x):
	return (x==255).all()*255

arr = ndimage.generic_filter(grey, func, size=(2, 2))
print(arr)

blank_image = np.zeros((arr.shape[0], arr.shape[1], 3), np.uint8)
mask = np.array(arr * 255, dtype=np.uint8)
masked = np.ma.masked_where(mask <= 0, mask)

cv2.namedWindow('After watershed', cv2.WINDOW_NORMAL)
cv2.imshow('After watershed', arr)
cv2.waitKey(0)
cv2.destroyAllWindows()