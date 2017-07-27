from legacyCode.utility import *


def findendsjunctions(edge_im, contours):
    kernel = np.zeros((3, 3), np.uint8)
    visited = np.zeros_like(contours)

    for c in contours:
        for i in range(c.shape[0]):
            kernel[1, 1] = edge_im[c[i]]
            kernel[0, 0] = edge_im[c[i][0] - 1, c[i][1] - 1]
            kernel[0, 1] = edge_im[c[i][0] - 1, c[i][1]]
            kernel[0, 2] = edge_im[c[i][0] - 1, c[i][1] + 1]
            kernel[1, 0] = edge_im[c[i][0], c[i][1] - 1]
            kernel[1, 2] = edge_im[c[i][0], c[i][1] + 1]
            kernel[2, 0] = edge_im[c[i][0] + 1, c[i][1] - 1]
            kernel[2, 1] = edge_im[c[i][0] + 1, c[i][1]]
            kernel[2, 2] = edge_im[c[i][0] + 1, c[i][1] + 1]
            marked = np.count_nonzero(kernel)
            if marked >= 4: # junction






if __name__ == '__main__':
    kernel = np.zeros((3, 3), np.uint8)

    #