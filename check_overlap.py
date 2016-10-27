import numpy as np
import math
from scipy.spatial.distance import pdist, squareform


# Written 10/4/2016
# Check overlap between two lines
# according to types of triangle that 3 points of end points of the line make.
# ---------------
# 10/13 - Added angle function

def angle(x,y,z):
    # angle = @(x,y,z) acosd((y^2+z^2-x^2)/(2*y*z)) ;   % cosine law

    # The angles of the triangle if one knows the three sides: arccos((a^2 + b^2 - c^2) / (2ab))
    # print("x, y, z =", x, y, z, ",", math.degrees(math.acos((y**2 + z**2 - x**2)/(2*y*z + 0.0))))
    return math.degrees(math.acos((y**2 + z**2 - x**2)/(2*y*z + 0.0))) # / math.pi*180.0


def check_overlap(line1, line2):
    # Format of the lines - [y1, x1, y2, x2, length, slope]

    pt1 = np.array((line1[1], line1[0]))
    pt2 = np.array((line1[3], line1[2]))

    pt3 = np.array((line2[1], line2[0]))
    pt4 = np.array((line2[3], line2[2]))

    # Calculate straight-line (Euclidean) distance between points
    a = np.linalg.norm(pt1 - pt2)
    b = np.linalg.norm(pt2 - pt3)
    c = np.linalg.norm(pt1 - pt3)
    d = np.linalg.norm(pt1 - pt4)
    e = np.linalg.norm(pt2 - pt4)
    f = np.linalg.norm(pt3 - pt4)

    # print("pt1", pt1, ",pt2", pt2, ",pt3", pt3, ",pt4", pt4)
    # print(a, b, c, d, e, f)

    a143 = angle(c, d, f)
    a134 = angle(d, c, f)

    a243 = angle(b, e, f)
    a234 = angle(e, b, f)

    a312 = angle(b, a, c)
    a321 = angle(c, b, a)

    a412 = angle(e, a, d)
    a421 = angle(d, a, e)

    # Print this in MATLAB
    # print("\na143", a143, "a134", a134, "|a243", a243, "a234", a234, "|a312", a312, "a321", a321, "|a412", a412, "a421", a421)

    # Return True if the lines could overlap
    return ((a143 < 90) & (a134 < 90)) | ((a243 < 90) & (a234 < 90)) | ((a312 < 90) & (a321 < 90)) | (
    (a412 < 90) & (a421 < 90))
