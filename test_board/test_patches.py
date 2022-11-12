from utils import data_utils
import numpy as np
from matplotlib import pyplot as plt

# 切片
# img_list = np.load(r"F://COVID/train/img.npy")
# mask_list = np.load(r"F://COVID/train/mask.npy")
#
# print(img_list.shape, mask_list.shape)
#
# mask_list = mask_list.reshape((mask_list.shape[0], mask_list.shape[1], mask_list.shape[2], -1))
#
# (patches_x, patches_y) = data_utils.generator_normal_patches(img_list, mask_list, 224)
#
# print(patches_x.shape, patches_y.shape)
#
# np.save("patches_x.npy", patches_x[0:81])
# np.save("patches_y.npy", patches_y[0:81])


# 拼接
patches_x = np.load(r"../y_pre_raw.npy")
x = data_utils.patches_to_img(patches_x, 224, (1, 672, 672, 1))
x = x[:, 0:505]
for i in range(x.shape[0]):
    plt.figure("Image{}".format(i))
    # plt.imshow(pre[i], cmap ='gray')
    plt.imshow(x[i], cmap='gray')
plt.show()
