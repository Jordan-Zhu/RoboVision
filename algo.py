
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology

from drawlinefeature import DrawLineFeature,drawconvex#zc
from lineseg import lineseg
from drawedgelist import drawedgelist
from Lseg_to_Lfeat_v2 import Lseg_to_Lfeat_v2   #zc
from LabelLineCurveFeature_v2 import LabelLineCurveFeature_v2  #zc
from merge_lines_v2 import merge_lines# zc
from LabelLineCurveFeature import classify_curves
from zeroElimMedianHoleFill import zeroElimMedianHoleFill


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def showimg(img, im_name='image', type='cv', write=False, imagename='img.png'):
    if type == 'plt':
        plt.figure()
        plt.imshow(img, im_name, interpolation='nearest', aspect='auto')
        # plt.imshow(img, 'gray', interpolation='none')
        plt.title(im_name)
        plt.show()

        if write:
            plt.savefig(imagename, bbox_inches='tight')
    elif type == 'cv':
        cv2.imshow(im_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if write:
            cv2.imwrite("../../images/%s", imagename, img)


def create_img(mat):
    blank_image = np.zeros((mat.shape[0], mat.shape[1], 3), np.uint8)
    # print(blank_image.shape)
    mask = np.array(mat * 255, dtype=np.uint8)
    masked = np.ma.masked_where(mask <= 0, mask)

    # plt.figure()
    # plt.imshow(blank_image, 'gray', interpolation='none')
    # plt.imshow(masked, 'gray_r', interpolation='none', alpha=1.0)
    # plt.title('canny + morphology')
    # plt.savefig('foo.png', bbox_inches='tight')
    # plt.show()

    return mask

def grad_dir(img):
    # compute x and y derivatives
    # OpenCV's Sobel operator gives better results than numpy gradient
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

    # calculate gradient direction angles
    # phase needs 64-bit input
    angle = cv2.phase(sobelx, sobely)

    # truncates number
    gradir = np.fix(180 + angle)

    return gradir


# Contrast Limited Adaptive Histogram Equalization
# Improves the contrast of our image.
def clahe(img, iter=1):
    for i in range(0, iter):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img


def normalize_depth(depthimg, colormap=False):
    # Normalize depth image
    min, max, minloc, maxloc = cv2.minMaxLoc(depthimg)
    adjmap = np.zeros_like(depthimg)
    dst = cv2.convertScaleAbs(depthimg, adjmap, 255 / (max - min), -min)
    if colormap == True:
        return cv2.applyColorMap(dst, cv2.COLORMAP_JET)
    else:
        return dst


def morpho(img):
    # kernel for dilation
    kernel = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    skel = morphology.skeletonize(dilation > 0)

    return skel


def edge_detect(depth):
    # Gradient of depth img
    graddir = grad_dir(depth)
    # Threshold image to get it in the RGB color space
    dimg1 = (((graddir - graddir.min()) / (graddir.max() - graddir.min())) * 255.9).astype(np.uint8)
    # Eliminate salt-and-pepper noise
    median = cv2.medianBlur(dimg1, 9)
    # Further remove noise while keeping edges sharp
    blur = cv2.bilateralFilter(median, 9, 25, 25)
    dimg1 = auto_canny(blur)
    skel1 = morpho(dimg1)
    showimg(create_img(skel1))
    # cnt1 = find_contours(create_img(skel1))

    # Depth discontinuity
    depthimg = normalize_depth(depth)
    dimg2 = clahe(depthimg, iter=2)
    showimg(dimg2)
    dimg2 = auto_canny(dimg2)
    skel2 = morpho(dimg2)
    # cnt2 = find_contours(create_img(skel2))

    # combine both images
    dst = (np.logical_or(skel1, skel2)).astype('uint8')
    dst = create_img(dst)
    return dst


def find_contours(im, mode=cv2.RETR_CCOMP):
    # im = cv2.imread('circle.png')
    # error: (-215) scn == 3 || scn == 4 in function cv::ipp_cvtColor
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    if mode == cv2.RETR_CCOMP:
        im2, contours, hierarchy = cv2.findContours(im, mode, cv2.CHAIN_APPROX_SIMPLE)
        newcontours = []
        for i in range(len(contours)):
            if hierarchy[0][i, 2] < 0:
                newcontours.append(contours[i])

        cv2.drawContours(im, newcontours, 2, (0, 255, 0), 1)
        return newcontours
    else:
        im2, contours, hierarchy = cv2.findContours(im, mode, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL cv2.RETR_CCOMP

    # show contours
    # cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
    #
    # # Display the image.
    # cv2.imshow("contours", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return contours


if __name__ == '__main__':
    # second argument is a flag which specifies the way
    # an image should be read. -1 loads image unchanged with alpha channel
    depthimg = cv2.imread('img/learn15.png', -1)
    colorimg = cv2.imread('img/clearn17.png', 0)

    showimg(normalize_depth(depthimg, colormap=True), 'depth')

    id = depthimg[100:, 100:480]  ## zc crop the region of interest

    siz = id.shape  ## image size of the region of interest
    thresh_m = 10
    label_thresh = 11
    # edges = edge_detect(depthimg, colorimg)
    edges = edge_detect(id)  # zc

    showimg(edges)
    # showimg(cntr1)
    # showimg(cntr2)

    cntrs = np.asarray(find_contours(edges))

    seglist = lineseg(cntrs, tol=2)
    drawedgelist(seglist, rowscols=[])

    # Get line features for later processing
    [linefeature, listpoint] = Lseg_to_Lfeat_v2(seglist, cntrs, siz)


    [line_new, listpoint_new, line_merged] = merge_lines(linefeature, listpoint, thresh_m, siz)

    # line_new = LabelLineCurveFeature_v2(depthimg, line_new, listpoint_new, label_thresh)
    line_new = classify_curves(depthimg, line_new, listpoint_new, label_thresh)
    # line_new = LabelLineCurveFeature_v2(depthimg, line_new, listpoint_new, label_thresh)
    DrawLineFeature(linefeature,siz,'line features')
    drawconvex(line_new, siz, 'convex')

    # TO-DO
    # - Check LabelLineCurveFeature_v2 with python input
    # - Write a function to make a window mask
    # - (Section 4) Take non-zero curve features and sort by angle (index 7 in MATLAB)

    # Long-term, to make the program easier to read and use
    # *- Create a Line object with properties: start, end, object/background, curvature/discontinuity
    #   distance from another line, check if line is overlapping
