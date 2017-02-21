import cv2
import numpy as np


def test_convexity(line_feature, img):
    blank_image = np.zeros_like(img, dtype=np.uint8)
    for i, e in enumerate(line_feature):
        print(e.shape)
        if cv2.isContourConvex(e):
            print('True')
            print(e)
            for i in range(1, e.shape[0]):
                # Reshape array for reading points
                print(e[i - 1][0])
                start = e[i][:, 0]
                # print(start)
                end = e[i][:, 1]
                # print(end)
            # cv2.drawContours(blank_image, e, 2, (0, 255, 0), 1)
                print('start', e[i - 1][0][0], e[i - 1][0][1], 'end', e[i][0])
                x1 = e[i - 1][0][0]
                y1 = e[i - 1][0][1]
                x2 = e[i][0][0]
                y2 = e[i][0][1]
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.namedWindow('Convex lines', cv2.WINDOW_NORMAL)
    cv2.imshow('Convex lines', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()