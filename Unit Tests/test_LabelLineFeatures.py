import scipy.io as sio
import numpy as np

from LabelLineFeatures import label_line_features


if __name__ == '__main__':
    edgeimg = sio.loadmat('lableline_in_DE1.mat')  # canny edge detect on depth image
    depthimg = sio.loadmat('lableline_in_Id.mat')  # depth image
    seglist = sio.loadmat('lableline_in_Line_new.mat')
    matlab_out = sio.loadmat('lableline_out_Line_new.mat')
    DE1 = edgeimg['DE1']
    Id = depthimg['Id']
    l2 = list(seglist['Line_new'])
    lout = matlab_out['Line_new']
    P = sio.loadmat('Parameter.mat')
    parameter = P['P']

    l = label_line_features(Id, DE1, l2, parameter)
    ll = l
    print(ll[0])
    col10 = ll[:][:, 10]
    col11 = ll[:, 11]
    col10out = lout[:, 10]
    col11out = lout[:, 11]
    extline = []
    lostline = []
    for i in range(len(col10)):
        if col10[i] != col10out[i]:
            extline += [i]
    for i in range(len(col11)):
        if col11[i] != col11out[i]:
            lostline += [i]

    print(len(extline))
    print(len(lostline))