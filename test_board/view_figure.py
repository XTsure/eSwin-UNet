import numpy as np
from matplotlib import pyplot as plt

pre = np.load('../pre.npy')
pre[pre > 0.1] = 1
pre[pre <= 0.1] = 0
# train_out = np.load('../train_out.npy').reshape((-1, 512, 512))
print(pre.shape)
pre = pre.reshape((-1, 512, 512))
print(pre.shape)

for i in range(15):
    plt.figure("Image{}".format(i))
    # plt.imshow(pre[i], cmap ='gray')
    plt.imshow(pre[i], cmap='gray')
plt.show()
