from PIL import Image
import numpy as np

train_img = np.load("F://COVID/train/img.npy")
train_mask = np.load("F://COVID/train/mask.npy")
train_lung_mask = np.load("F://COVID/train/lung_mask.npy")
print(train_img.shape, train_mask.shape, train_lung_mask.shape)

train_img_list = []
train_mask_list = []
train_lung_mask_list = []
for i in range(train_img.shape[0]):
    ti = Image.fromarray(train_img[i])
    tm = Image.fromarray(train_mask[i])
    tlm = Image.fromarray(train_lung_mask[i])

    ti = ti.resize((224, 224), Image.ANTIALIAS)
    tm = tm.resize((224, 224), Image.ANTIALIAS)
    tlm = tlm.resize((224, 224), Image.ANTIALIAS)

    train_img_list.append(np.array(ti))
    train_mask_list.append(np.array(tm))
    train_lung_mask_list.append(np.array(tlm))

train_img_r = np.array(train_img_list)
train_mask_r = np.array(train_mask_list)
train_lung_mask_r = np.array(train_lung_mask_list)

print(train_img.shape, train_mask.shape, train_lung_mask.shape)
print(train_img_r.shape, train_mask_r.shape, train_lung_mask_r.shape)

np.save("F://COVID_224/train/img.npy", train_img_r)
np.save("F://COVID_224/train/mask.npy", train_mask_r)
np.save("F://COVID_224/train/lung_mask.npy", train_lung_mask_r)


test_img = np.load("F://COVID/test/img.npy")
test_mask = np.load("F://COVID/test/mask.npy")
test_lung_mask = np.load("F://COVID/test/lung_mask.npy")
print(test_img.shape, test_mask.shape, test_lung_mask.shape)

test_img_list = []
test_mask_list = []
test_lung_mask_list = []
for i in range(test_img.shape[0]):
    ti = Image.fromarray(test_img[i])
    tm = Image.fromarray(test_mask[i])
    tlm = Image.fromarray(test_lung_mask[i])

    ti = ti.resize((224, 224), Image.ANTIALIAS)
    tm = tm.resize((224, 224), Image.ANTIALIAS)
    tlm = tlm.resize((224, 224), Image.ANTIALIAS)

    test_img_list.append(np.array(ti))
    test_mask_list.append(np.array(tm))
    test_lung_mask_list.append(np.array(tlm))

test_img_r = np.array(test_img_list)
test_mask_r = np.array(test_mask_list)
test_lung_mask_r = np.array(test_lung_mask_list)

print(test_img.shape, test_mask.shape, test_lung_mask.shape)
print(test_img_r.shape, test_mask_r.shape, test_lung_mask_r.shape)

np.save("F://COVID_224/test/img.npy", test_img_r)
np.save("F://COVID_224/test/mask.npy", test_mask_r)
np.save("F://COVID_224/test/lung_mask.npy", test_lung_mask_r)