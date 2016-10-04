import numpy as np

# Written 10/4/2016
# Check overlap between two lines
# according to types of triangle that 3 points of end points of the line
# make.


def check_overlap(line1, line2):
    # LineFeature = [y1 x1 y2 x2 L m] ;

    pt1 = np.array(line1[0], line1[1])
    pt2 = np.array(line1[2], line1[3])

    pt3 = np.array(line2[0], line2[1])
    pt4 = np.array(line2[2], line2[3])

    # Use Euclidean distance to calculate
    # distance of 3D points
    a = np.linalg.norm(pt1 - pt2)
    b = np.linalg.norm(pt2 - pt3)
    c = np.linalg.norm(pt1 - pt3)
    d = np.linalg.norm(pt1 - pt4)
    e = np.linalg.norm(pt2 - pt4)
    f = np.linalg.norm(pt3 - pt4)

    # angle = @(x,y,z) acosd((y^2+z^2-x^2)/(2*y*z)) ;   % cosine law

    a143 = np.angle((c, d, f))
    a134 = np.angle((d, c, f))

    a243 = np.angle((b, e, f))
    a234 = np.angle((e, b, f))

    a312 = np.angle((b, a, c))
    a321 = np.angle((c, b, a))

    a412 = np.angle((e, a, d))
    a421 = np.angle((d, a, e))

    # Return f_val
    return ((a143 < 90) & (a134 < 90)) | ((a243 < 90) & (a234 < 90)) | ((a312 < 90) & (a321 < 90)) | (
    (a412 < 90) & (a421 < 90))
