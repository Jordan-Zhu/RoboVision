import numpy as np
import math
from scipy.spatial.distance import pdist, squareform


# Check overlap between two lines
# according to types of triangle that 3 points of end points of the line make.
# ---------------


def angle_between(x,y,z):
    # angle = @(x,y,z) acosd((y^2+z^2-x^2)/(2*y*z)) ;   % cosine law

    # The angles of the triangle if one knows the three sides: arccos((a^2 + b^2 - c^2) / (2ab))
    # print("x, y, z =", x, y, z)
    # print(math.degrees(math.acos((y**2 + z**2 - x**2)/(2*y*z + 0.0))))
    cos_ang = (y**2 + z**2 - x**2) / (2*y*z)
    tol = 5e-6
    if math.fabs(cos_ang) > 1.0:
        if math.fabs(cos_ang) - 1.0 < tol:
            cos_ang = math.modf(cos_ang)[1]
        else:
            raise ValueError('Invalid arguments (vectors not normalized?)')
    return math.degrees(math.acos(cos_ang))

    # t1 = y**2 + z**2 - x**2
    # t2 = 2*y*z
    # print('t1 =', t1, 't2 =', t2, 't1/t2 =', t1/t2, 'acos =', math.acos(1.0))
    # print('acos =', math.acos(t1/t2))
    # return math.degrees(math.acos((y**2 + z**2 - x**2)/(2*y*z + 0.0))) # / math.pi*180.0
    # return 0 if t1 < 0 else math.degrees(math.acos(t1 / t2))


def check_overlap(line1, line2):
    # Format of the lines - [y1, x1, y2, x2, length, slope]

    pt1 = np.array([line1[1], line1[0]]).astype(int)
    pt2 = np.array([line1[3], line1[2]]).astype(int)

    pt3 = np.array([line2[1], line2[0]]).astype(int)
    pt4 = np.array([line2[3], line2[2]]).astype(int)

    # print('pt1 =', pt1, 'pt2 =', pt2, 'pt3 =', pt3, 'pt4 =', pt4)

    # Calculate straight-line (Euclidean) distance between points
    a = np.linalg.norm(pt1 - pt2)
    b = np.linalg.norm(pt2 - pt3)
    c = np.linalg.norm(pt1 - pt3)
    d = np.linalg.norm(pt1 - pt4)
    e = np.linalg.norm(pt2 - pt4)
    f = np.linalg.norm(pt3 - pt4)

    # print("pt1", pt1, ",pt2", pt2, ",pt3", pt3, ",pt4", pt4)
    # print(a, b, c, d, e, f)

    a143 = angle_between(c, d, f)
    # print('a143 =', a143)
    a134 = angle_between(d, c, f)
    # print('a134 =', a134)

    a243 = angle_between(b, e, f)
    # print('a243 =', a243)
    a234 = angle_between(e, b, f)
    # print('a234 =', a234)

    a312 = angle_between(b, a, c)
    # print('a312 =', a312)
    a321 = angle_between(c, b, a)
    # print('a321 =', a321)

    a412 = angle_between(e, a, d)
    # print('a412 =', a412)
    a421 = angle_between(d, a, e)
    # print('a421 =', a421)

    # Print this in MATLAB
    # print('a312 = ', a312, '\n')
    # print("\na143", a143, "a134", a134, "|a243", a243, "a234", a234, "|a312", a312, "a321", a321, "|a412", a412, "a421", a421)

    # Return True if the lines could overlap
    return ((a143 < 90) & (a134 < 90)) | ((a243 < 90) & (a234 < 90)) | ((a312 < 90) & (a321 < 90)) | (
    (a412 < 90) & (a421 < 90))
