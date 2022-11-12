import numpy as np
from matplotlib import pyplot as plt

train_x = np.load('../train_x.npy')
print(train_x.shape)
print(train_x)

# # train_out = np.load('../train_out.npy').reshape((-1, 512, 512))
# pre = pre.reshape((-1, 224, 224))
# print(pre.shape)
#
#
#
for i in range(train_x.shape[0]):
    plt.figure("Image{}".format(i))
    # plt.imshow(pre[i], cmap ='gray')
    plt.imshow(train_x[i])
plt.show()