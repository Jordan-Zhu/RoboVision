import numpy as np


cp = np.load("saveCntr.npy")
print(cp)
# x = []
# y = []
# z = []
# for i in range(cp.shape[0]):
#     x.append(np.vstack(cp[i])[:, 0])
#     y.append(np.vstack(cp[i])[:, 1])
#     z.append(np.vstack(cp[i])[:, 2])
# x_val = np.hstack(np.array(x))
# y_val = np.hstack(np.array(y))
# z_val = np.hstack(np.array(z))
np.save("save_cX", cp[:, 0])
np.save("save_cY", cp[:, 1])
np.save("save_cZ", cp[:, 2])
# x, y, z = cp[0, 0]

pairs = np.load("save_pairs.npy")
print(pairs)
print(pairs[:, 0])
np.save("save_pairsX", pairs[:, 0])
np.save("save_pairsY", pairs[:, 1])
np.save("save_pairsZ", pairs[:, 2])
