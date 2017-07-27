import cv2


if __name__ == '__main__':
    for numImg in [3]:
        im = cv2.imread('img/clearn%d.png' % numImg, -1)

        # fromCenter = False
        r = cv2.selectROI(im, fromCenter=False)

        # Crop image
        imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # Display cropped image
        cv2.imshow("cropped", imCrop)
        cv2.waitKey(0)
