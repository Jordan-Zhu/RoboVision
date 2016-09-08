import cv2
import numpy as np
import math

# I = cv2.imread('C:\OSD-data\learn1.png',-1)
I = cv2.imread('img/learn1.png', -1)
def zeroElimMedianFilter (I):
    dim = I.shape
    rows = dim[0]
    cols = dim[1]
    I = np.lib.pad(I, ((2, 2),(2,2)) ,'edge')
    k = -2
    y = 2
    R = np.zeros((rows,cols))
    for m in range (0,rows-1):
        for n in range (0,cols-1):
            j=k+m+3   
            n=y+m+3         
            A = I[j-1:n, j-1:n]
            dimA = A.shape
            rowsA = dimA[0]
            colsA = dimA[1]
            A = np.reshape(A,(1,colsA*rowsA))
            A=A[A !=0]
            if A.size > 0:
                A = np.sort(A)
                dimA = A.size
                R[m, n] = A[(math.ceil(dimA/2)-1)]
            else:
                R[m,n] = I[m,n]
    return R
B = zeroElimMedianFilter (I)
print(B)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', B)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
            
    
