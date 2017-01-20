import math
import numpy as np


def Lseg_to_Lfeat(ListSegments, ListEdges, imgsize):
	length = len(ListSegments)

	LineFeature = []
	ListPoint = []
	for i in range(0, length):
		curr = ListSegments[i]
		lineLen = len(curr)

		for j in range(0, lineLen):
			x1 = curr[c2, 0].astype(int)
            x2 = curr[c2+1, 0].astype(int)
            y1 = curr[c2, 1].astype(int)
            y2 = curr[c2+1, 1].astype(int)

            slope = round((y2-y1)/float((x2 - x1)), 2)
            pix1 = np.ravel_multi_index((y1, x1), imgsize, order='F')
            pix2 = np.ravel_multi_index((y2, x2), imgsize, order='F')
            lineLength = round(math.sqrt(round((x2 - x1) ** 2 + (y2 - y1) ** 2, 2)), 2)
            alpha = round(math.atan(-m) * 180 / math.pi, 2)
            
            LineFeature