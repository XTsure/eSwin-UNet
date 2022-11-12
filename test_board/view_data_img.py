import numpy as np
from matplotlib import pyplot as plt

img = np.load('F://COVID_224/train/img.npy')
# pre = pre.reshape((-1, 224, 224))
# print(pre.shape)

for i in range(img.shape[0]):
    plt.figure("Image{}".format(i))
    plt.imshow(img[i], cmap='gray')
plt.show()
