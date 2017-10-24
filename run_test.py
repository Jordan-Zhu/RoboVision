import numpy as np
from skimage.measure import LineModelND, ransac
from PlaneModelC import PlaneModelND


cntr_pairs = np.load("saveCntr.npy")

idx_col = np.load("save_mX.npy")
idx_row = np.load("save_mY.npy")
idx_z = np.load("save_mZ.npy")

# idx_row = np.random.randint(200, size=(10, 1))
# idx_col = np.random.randint(200, size=(10, 1))

# pointcloud = np.random.randint(500, size=(480, 640, 3))
# pointcloud[:, :, 2] = 0

pointcloud = []
for i in range(idx_col.shape[0]):
    temp = [idx_col[i], idx_row[i], idx_z[i]]
    pointcloud.append(temp)

# data1 = pointcloud[idx_row, idx_col, :]
data1 = pointcloud
# print(data1)
data1 = np.squeeze(data1)
# print(data1.shape)
# print(data1)

# robustly fit plane and line only using inlier data with RANSAC algorithm
model_robustP, inliersP = ransac(data1, PlaneModelND,  min_samples=3, residual_threshold=1, max_trials=20)

new_pc = []
for i in range(inliersP.shape[0]):
    if inliersP[i] == True:
        # print("i = ", i)
        new_pc.append(pointcloud[i])
print("new pc", new_pc)
# np.save("new_pc.npy", new_pc)
new_x = []
new_y = []
new_z = []
for i in range(len(new_pc)):
    new_x.append(new_pc[i][0])
    new_y.append(new_pc[i][1])
    new_z.append(new_pc[i][2])

np.save("new_x.npy", new_x)
np.save("new_y.npy", new_y)
np.save("new_z.npy", new_z)
# print("inliers", inliersP)
# print("Params", model_robustP.params)




