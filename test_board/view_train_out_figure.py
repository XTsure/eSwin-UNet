import numpy as np
from matplotlib import pyplot as plt

train_out = np.load('../train_out.npy')
print(train_out.shape)
train_out = train_out.reshape((-1, 224, 224))
print(train_out.shape)

train_out[train_out > 0.1] = 1
train_out[train_out <= 0.1] = 0

for i in range(4):
    plt.figure("Image{}".format(i))
    # plt.imshow(pre[i], cmap ='gray')
    plt.imshow(train_out[i], cmap='gray')
plt.show()
