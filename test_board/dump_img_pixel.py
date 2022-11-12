import numpy as np
from matplotlib import pyplot as plt

pre = np.load('../pre0.npy')
# train_out = np.load('../train_out.npy').reshape((-1, 512, 512))
pre = pre.reshape((-1, 512, 512))
print(pre.shape)

for i in pre:
    for j in i:
        for m in j:
            if not m == 255:
                print(m)
