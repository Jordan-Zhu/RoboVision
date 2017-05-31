import cv2
import numpy as np
from utility import get_orientation, get_ordering, normalize_depth, get_direction_py, draw_lf


def swap_indices(arr):
    res = []
    for i, e in enumerate(arr):
        res.append([arr[i][1] - 1, arr[i][0] - 1])
    return np.array(res)


def roipoly(src, poly):
    mask = np.zeros_like(src, dtype=np.uint8)

    overlay = normalize_depth(src, colormap=True)
    output = normalize_depth(src, colormap=True)
    alpha = 0.5

    cv2.fillConvexPoly(mask, poly, 255)  # Create the ROI
    cv2.fillConvexPoly(overlay, poly, (255, 255, 255))
    cv2.putText(overlay, "ROI Poly: alpha={}".format(alpha), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    # cv2.line(output, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 255, 255), 3)
    res = src * mask
    # print('mask1 count', np.count_nonzero(mask))
    # cv2.namedWindow('roi poly', cv2.WINDOW_NORMAL)
    # cv2.imshow('roi poly', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res


def classify_curves(src, list_lines, list_points, window_size):
    im_size = src.shape
    out = []
    for index, line in enumerate(list_lines):
        # print(line)
        pt1, pt2, pt3, pt4 = get_direction_py(line, window_size)
        img = normalize_depth(src, colormap=True)


        cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 0, 255), 1)
        cv2.line(img, (int(pt3[0]), int(pt3[1])), (int(pt4[0]), int(pt4[1])), (0, 0, 255), 1)
        cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 0, 0), 1)

        # cv2.namedWindow('Points', cv2.WINDOW_NORMAL)
        # cv2.imshow('Points', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        win = np.array(get_ordering(pt1, pt2, pt3, pt4))

        # win2 = get_ordering(pt1, pt2, pt3, pt4)
        # win = swap_indices(win)

        # print(win)
        mask4 = roipoly(src, win)
        # mask0 = np.array(roipoly2(src, line, win2))
        area = cv2.contourArea(np.int32([win]))
        # print('Mask4 area:', area)
        # print('Mask4 count:', cv2.countNonZero(mask4))


        # a1 = np.mean(mask4[np.nonzero(mask4)])
        a1 = 0 if cv2.countNonZero(mask4) == 0 else sum(src[np.nonzero(mask4)]) / cv2.countNonZero(mask4)
        # a1 = sum(mask4)) / cv2.countNonZero(np.array(mask4))

        # Get mask of values on the line
        lx = list_points[index]
        print(lx)
        temp_list = []
        for ii in lx:
            x1, y1 = np.unravel_index([ii], im_size, order='C')
            # print(x1, ",", y1)
            temp_list.append((x1[0], y1[0]))

        # new_list = [int(i) for i in temp_list[0], int(j) for j in temp_list[1]]
        temp_list = np.squeeze(np.array(temp_list))
        print(temp_list)
        new_list = np.vstack(([temp_list[0].T], [temp_list[1].T])).T
        # print(new_list)
        # points = np.array([[164, 65], [164, 544], [165, 385], [164, 546]])
        # for i in range(1, 10):
        #     cv2.line(img, (points[i][0], points[i][1]), (points[i]))
        cv2.fillConvexPoly(img, new_list, (255,255,255))
        # #
        # cv2.namedWindow('Points', cv2.WINDOW_NORMAL)
        # cv2.imshow('Points', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        mask5 = []
        for i in temp_list:
            print(i[0], i[1])
            mask5.append(src[i[0], i[1]])

        print('mask5', mask5)
        # mask5 = [value for value in mask5 if value != 0]
        a2 = 0 if not mask5 else np.mean(mask5)

        # print('A1', a1, '\nA2', a2, '\n')

        b1 = cv2.countNonZero(mask4) * a1 - len(mask5) * a2
        try:
            b11 = float(b1) / (cv2.countNonZero(mask4) - len(mask5))
        except ZeroDivisionError:
            b11 = float('nan')

        # print('B11', b11)
        if b11 < a2:
            # print('IF\n\n')
            out.append(np.append(list_lines[index], [12]))
        else:
            # print('ELSE\n\n')
            out.append(np.append(list_lines[index], [13]))

    return np.asarray(out)
