import cv2
import matplotlib as plt
import numpy as np
import random as rand
from skimage import morphology


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


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


def curve_discont(depth_im):
    # Gradient of depth img
    graddir = grad_dir(depth_im)
    # Threshold image to get it in the RGB color space
    dimg1 = (((graddir - graddir.min()) / (graddir.max() - graddir.min())) * 255.9).astype(np.uint8)
    # Eliminate salt-and-pepper noise
    median = cv2.medianBlur(dimg1, 9)
    # Further remove noise while keeping edges sharp
    blur = cv2.bilateralFilter(median, 9, 25, 25)
    dimg1 = auto_canny(blur)
    skel1 = morpho(dimg1)
    # showimg(create_img(skel1), "Morphology + canny on depth image")
    cnt1 = find_contours(create_img(skel1), cv2.RETR_EXTERNAL)

    return skel1, cnt1


def depth_discont(depth_im):
    # Depth discontinuity
    depthimg = normalize_depth(depth_im)
    dimg2 = clahe(depthimg, iter=2)
    dimg2 = auto_canny(dimg2)
    skel2 = morpho(dimg2)
    # showimg(dimg2, "Depth discontinuity w/ tone balancing")
    cnt2 = find_contours(create_img(skel2), cv2.RETR_EXTERNAL)

    return skel2, cnt2


def edge_detect(depth):
    curve_disc, curve_con = curve_discont(depth)
    depth_disc, depth_con = depth_discont(depth)

    # squeeze_ndarr(curve_con)
    # squeeze_ndarr(depth_con)

    # combine both images
    dst = (np.logical_or(curve_disc, depth_disc)).astype('uint8')
    dst = create_img(dst)
    # showimg(dst, "Depth + Discontinuity")
    skel_dst = morpho(dst)
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
    sqz_contours(res)
    # squeeze_ndarr(res)

    dst = find_contours(create_img(skel_dst), cv2.RETR_EXTERNAL)
    # showimg(dst, "Depth + Discontinuity")


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


def find_contours(im, mode=cv2.RETR_CCOMP):
    # im = cv2.imread('circle.png')
    # error: (-215) scn == 3 || scn == 4 in function cv::ipp_cvtColor
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    height = im.shape[0]
    width = im.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)
    img = normalize_depth(im, colormap=True)
    if mode == cv2.RETR_CCOMP:
        im2, contours, hierarchy = cv2.findContours(im, mode, cv2.CHAIN_APPROX_NONE)
        newcontours = []
        for i in range(len(contours)):
            if hierarchy[0][i, 2] < 0:
                # color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
                # cv2.drawContours(blank_image, contours, i, color, 1, 8)
                newcontours.append(contours[i])

        # Display contours
        # draw_contours(blank_image, contours)
        # cv2.imshow("CONTOURS", blank_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cntrs = np.array(newcontours)
        # print(cntrs)
        sqz_contours(cntrs)

        return cntrs
    else:
        im2, contours, hierarchy = cv2.findContours(im, mode, cv2.CHAIN_APPROX_NONE)
        # draw_contours(blank_image, contours)
    # cv2.RETR_EXTERNAL cv2.RETR_CCOMP

    cntrs = np.array(contours)
    sqz_contours(cntrs)

    return cntrs


def draw_contours(im, contours):
    height = im.shape[0]
    width = im.shape[1]
    overlay = np.zeros((height, width, 3), np.uint8)
    output = np.zeros((height, width, 3), np.uint8)
    alpha = 0.5

    # cv2.putText(overlay, "ROI Poly: alpha={}".format(alpha), (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    for i in range(len(contours)):
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.drawContours(im, contours, i, color, 1, 8)
        # cv2.imshow("contours", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # cv2.addWeighted(overlay, alpha, output, 1 - alpha,
    #                 0, output)
    cv2.imshow("contours", im)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


# XORs contours so new contours are preserved while maintaining the initial contours.
# def fix_contours(contours):
#
#     for i in range(len(contours)):



# Squeezes the array and swaps the columns to match Numpy's col, row ordering
def sqz_contours(contours):
    squeeze_ndarr(contours)
    # ADVANCED SLICING
    for i in range(contours.shape[0]):
        swap_cols(contours[i], 0, 1)

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
        cv2.namedWindow(im_name, cv2.WINDOW_NORMAL)
        cv2.imshow(im_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if write:
            cv2.imwrite("../../images/%s", imagename, img)


def roipoly(src, line, poly):
    mask = []
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])
    dyy = line[0] - line[2]
    dxx = line[1] - line[3]

    if dy > dx or dy == dx:
        xfp = min(int(poly[0][1]), int(poly[1][1])) if dxx * dyy > 0 else max(int(poly[0][1]), int(poly[1][1]))
        mask_len = int(poly[3][1] - poly[0][1])
        y_range_start = min(int(poly[0][0]), int(poly[1][0]))
        y_range_end = max(int(poly[0][0]), int(poly[1][0]))

        for i in range(y_range_start, y_range_end):
            x0 = int(round(xfp))
            mask += list(src[i, x0:x0 + mask_len])
            step = (poly[1][1] - poly[0][1] + 0.0) / (poly[1][0] - poly[0][0] + 0.0)
            xfp += step
    else:
        yfp = min(int(poly[0][0]), int(poly[1][0])) if dxx * dyy > 0 else max(int(poly[0][0]), int(poly[1][0]))
        mask_len = int(poly[3][0] - poly[0][0])
        x_range_start = min(int(poly[0][1]), int(poly[1][1]))
        x_range_end = max(int(poly[0][1]), int(poly[1][1]))

        for i in range(x_range_start, x_range_end):
            y0 = int(round(yfp))
            mask += list(src[y0:y0 + mask_len, i])
            step = (poly[1][0] - poly[0][0] + 0.0) / (poly[1][1] - poly[0][1] + 0.0)
            yfp += step
    return mask


def get_orientation(line, window_size):
    dy = abs(line[0] - line[2])
    dx = abs(line[1] - line[3])
    # Vertical or horizontal line test
    if dy > dx or dy == dx:
        pt1 = [line[0], line[1] - window_size]
        pt2 = [line[0], line[1] + window_size]
        pt3 = [line[2], line[3] - window_size]
        pt4 = [line[2], line[3] + window_size]
        return pt1, pt2, pt3, pt4
    else:
        pt1 = [line[0] - window_size, line[1]]
        pt2 = [line[0] + window_size, line[1]]
        pt3 = [line[2] - window_size, line[3]]
        pt4 = [line[2] + window_size, line[3]]
        return pt1, pt2, pt3, pt4


def get_direction_py(line, window_size):
    dx = abs(line[0] - line[2])
    dy = abs(line[1] - line[3])
    # Vertical line test
    if dy > dx or dy == dx:
        pt1 = [line[0] - window_size, line[1]]
        pt2 = [line[0] + window_size, line[1]]
        pt3 = [line[2] - window_size, line[3]]
        pt4 = [line[2] + window_size, line[3]]
        return pt1, pt2, pt3, pt4
    else:
        pt1 = [line[0], line[1] - window_size]
        pt2 = [line[0], line[1] + window_size]
        pt3 = [line[2], line[3] - window_size]
        pt4 = [line[2], line[3] + window_size]
        return pt1, pt2, pt3, pt4

def get_ordering(pt1, pt2, pt3, pt4):
    temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
    temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
    res = np.array([pt1, pt3, pt4, pt2]) if temp1 > temp2 else np.array([pt1, pt4, pt3, pt2])
    return [[int(i) for i in pt] for pt in res]


def draw_convex(line_feature, img):
    # blank_image = np.zeros_like(img, dtype=np.uint8)
    # blank_image = normalize_depth(img)
    height = img.shape[0]
    width = img.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)


    # print(line_feature[0])
    for i, e in enumerate(line_feature):
        if e[10] == 13:
            x1 = int(e[1])
            y1 = int(e[0])
            x2 = int(e[3])
            y2 = int(e[2])
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # cv2.namedWindow('Convex lines', cv2.WINDOW_NORMAL)
            # cv2.imshow('Convex lines', blank_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    cv2.namedWindow('Convex lines', cv2.WINDOW_NORMAL)
    cv2.imshow('Convex lines', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_convex_py(line_feature, img):
    blank_image = normalize_depth(img)

    # print(line_feature[0])
    for i, e in enumerate(line_feature):
        if e[10] == 13:
            x1 = int(e[0])
            y1 = int(e[1])
            x2 = int(e[2])
            y2 = int(e[3])
            cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # cv2.namedWindow('Convex lines', cv2.WINDOW_NORMAL)
            # cv2.imshow('Convex lines', blank_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    cv2.namedWindow('Convex lines', cv2.WINDOW_NORMAL)
    cv2.imshow('Convex lines', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_lf(line_feature, img):
    # blank_image = normalize_depth(img, colormap=True)
    blank_image = np.zeros_like(img)

    # print(line_feature[0])
    for i, e in enumerate(line_feature):
        x1 = int(e[0])
        y1 = int(e[1])
        x2 = int(e[2])
        y2 = int(e[3])
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.line(blank_image, (x1, y1), (x2, y2), color, 2)
        # cv2.namedWindow('Convex lines', cv2.WINDOW_NORMAL)
        # cv2.imshow('Convex lines', blank_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    cv2.namedWindow('Line features', cv2.WINDOW_NORMAL)
    cv2.imshow('Line features', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_lp(list_point, img, imgsize):

    for i, e in enumerate(list_point):
        for j, element in enumerate(e[0]):
            # print(e2)
            x, y = np.unravel_index([element], imgsize, order='C')
            cv2.line(img, (x, y), (x, y), (255, 0, 0), 2)

    cv2.namedWindow('List Points', cv2.WINDOW_NORMAL)
    cv2.imshow('List Points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def swap_indices_m(arr):
    res = []
    for i, e in enumerate(arr):
        res.append([arr[i][1] - 1, arr[i][0] - 1])
    return np.array(res)


def swap_indices(arr):
    res = []
    for i, e in enumerate(arr):
        res.append([arr[i][1], arr[i][0]])
    return np.array(res)


def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]


def squeeze_ndarr(arr):
    temp = []
    for i in range(arr.shape[0]):
        temp.append(np.squeeze(arr[i]))
    np.copyto(arr, np.array(temp))