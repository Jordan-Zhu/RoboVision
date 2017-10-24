import cv2
import numpy as np
import util as util

window_size = 3

def roipoly(src, poly):
    window = np.zeros_like(src, dtype=np.uint8)
    win = util.swap_indices(poly)
    cv2.fillConvexPoly(window, win, 255)  # Create the ROI
    return window

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


def get_ordering(pt1, pt2, pt3, pt4):
    temp1 = np.linalg.norm(np.subtract((np.add(pt1, pt3) / 2.0), (np.add(pt2, pt4) / 2.0)))
    temp2 = np.linalg.norm(np.subtract((np.add(pt1, pt4) / 2.0), (np.add(pt2, pt3) / 2.0)))
    res = np.array([pt1, pt3, pt4, pt2]) if temp1 > temp2 else np.array([pt1, pt4, pt3, pt2])
    return [[int(i) for i in pt] for pt in res]



if __name__ == "__main__":
    matched_lines = np.load("savePairs.npy")
    contours = np.load("saveCntr.npy")
    img = cv2.imread("test_im.png", -1)
    print("matched lines", matched_lines)
    # print("contours", contours)
    img2 = cv2.imread("test_im.png", -1)
    img2 = util.normalize_depth(img2)
    img2 = util.clahe(img2, iter=2)

    cy = round(img.shape[0] / 2)
    cx = round(img.shape[1] / 2)

    # USE COPY.DEEPCOPY if you don't want to edit variables passed in
    # AKA THE IMAGES, DON'T DO THIS
    P = {"path": 'outputImg\\',
         "img": img,
         "height": img.shape[0],
         "width": img.shape[1],
         "img_size": img.shape,
         "cx": cx,
         "cy": cy,
         "focal_length": 300,
         "thresh": 20,
         "window_size": 10,
         "min_len": 20,
         "min_dist": 10,
         "max_dist": 200,
         "delta_angle": 20
         }

    # Values that depend on values in P
    P2 = {"blank_image": np.zeros((P["height"], P["width"], 3), np.uint8)}

    # adds all these new values to P
    P.update(P2)

    # done every line
    points = []
    line = []
    for k in range(len(matched_lines)):
        for m in range(2):
            if matched_lines[k][m][10] == 13:
                line = matched_lines[k][m]
                pt1, pt2, pt3, pt4 = get_orientation(matched_lines[k][m], window_size)
                roi_win = np.array(get_ordering(pt1, pt2, pt3, pt4))
                window = roipoly(img, roi_win)

                points = np.where(window == 255)

                cv2.imshow("window", window)

                cv2.imshow("Sobel", util.auto_canny(img2))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # print(points[0].shape)
    # for i in range(points[0].shape[0]):
    #     print(points[0][i], end=" ")
    #     if i % (window_size * 2) == 0:
    #         print("\n")

    # print([img[points[0][i], points[1][i]] for i in points[0]])

    magn = []
    # Horizontal
    for i in range(0, points[0].shape[0], window_size * 2):
        done = False
        for j in range(i, i + window_size * 2 - 1):
            # print("i = ", i, "j = ", j)
            if j+1 == points[0].shape[0]:
                magn.append(0)
                done = True
                break
            magn.append(img[points[0][j]][points[1][j]] - img[points[0][j+1]][points[1][j+1]])
        if done == True:
            break
        magn.append(0)
    # index 12 is object lies on left/right
    print("magnitude", magn)
    mask = np.zeros(len(magn), dtype=bool)
    if line[12] == 2:
        for i in range(0, len(magn), window_size * 2):
            col = np.array(magn[i:i+window_size * 2 - 1])
            maxes = np.where(col == col.max())[0][-1]
            # print(maxes)
            mask[i + maxes] = True

    print(mask)
    new_pts = np.array([points[0][np.logical_and(points[0], mask)], points[1][np.logical_and(points[1], mask)]])
    print(new_pts)

    pc = []
    for i in range(new_pts[0].shape[0]):
        # print(new_pts[1][i], ",", new_pts[0][i])
        pc.append(util.depth_to_3d(new_pts[1][i], new_pts[0][i] + 4, P))

    # print(pc)
    # np.save("save_magn", np.array(pc))
    mX = []
    mY = []
    mZ = []
    for i in range(len(pc)):
        mX.append(pc[i][0])
        mY.append(pc[i][1])
        mZ.append(pc[i][2])
    np.save("save_mX", mX)
    np.save("save_mY", mY)
    np.save("save_mZ", mZ)
    # Vertical

    print(len(magn))
    # for i in range(0, len(magn), window_size * 2):
    #     for j in range(i, i + window_size * 2):
    #         print(magn[j], end = " ")
    #     print('\n')