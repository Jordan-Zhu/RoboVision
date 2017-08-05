import math
import numpy as np
import distance2d as d2


if __name__ == '__main__':
    # LineFeature = [y1 x1 y2 x2 L m];
    line1 = np.array([20, 10, 40, 10, 20, 90])
    line2 = np.array([20, 20, 50, 20, 30, 90])

    dist = d2.distance2d(line1, line2)
    print(dist, "dist")
