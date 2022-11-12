import sys
import os
import math
import random
import torch
import imageio
import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

from PIL import Image
from torchvision.transforms import functional as ttf
from utils import sys_utils


def nii_2_png(nii_path, png_path):
    """
    nii文件转换为png存储
    :param nii_path: nii文件路径
    :param png_path: png文件路径
    """

    # nii文件路径不存在，退出程序
    if not os.path.exists(nii_path):
        print('-------------路径{}不存在-------------'.format(nii_path))
    else:
        # img文件路径不存在，创建文件夹
        if not os.path.exists(png_path):
            os.makedirs(png_path)  # 新建文件夹

        # 加载nii文件
        nii = nib.load(nii_path)
        nii_data = nii.get_fdata()

        # 获取图片的维度
        (x, y, z) = nii.shape

        for i in range(z):
            # 选择切片方向
            nii_slice = nii_data[:, :, i]
            # 生成png图像，np.rot90(nii_slice)旋转图片，根据需要使用
            imageio.imwrite(os.path.join(png_path, '{}.png'.format(i)), np.rot90(nii_slice))


def img_2_npy(img_folder_path, npy_folder_path, file_name, mode="RGB"):
    """
    图片文件转npy文件
    :param img_folder_path: 图片存储文件夹路径
    :param npy_folder_path: npy文件路径
    :param file_name: 存储文件名
    :param mode: 图片读取格式
    """

    array_of_img = []

    for filename in os.listdir(img_folder_path):
        # 图片路径
        img_path = os.path.join(img_folder_path, filename)

        # 只处理png文件
        if ".png" in img_path:
            if "GRAY" == mode:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path)

            array_of_img.append(img)

    if not os.path.exists(npy_folder_path):
        os.makedirs(npy_folder_path)  # 新建文件夹

    # 存储npy文件
    np.save(os.path.join(npy_folder_path, file_name), array_of_img)


def load_data(data_path, file_name_list):
    """
    加载数据集（hdf5格式文件）
    :param data_path: 数据集存储路径
    :param file_name_list: 数据文件列表
    :return: 训练集和测试集 img, gt, lung_and_infection_mask, lung_mask
    """

    ex_str = "("
    for key, file_name in enumerate(file_name_list):
        # 文件路径
        file_path = os.path.join(data_path, file_name + ".npy")

        # 检查路径是否存在
        if os.path.exists(file_path):
            ex_str = ex_str + "np.load(r\"" + file_path + "\"), "
        else:
            print("---------------文件{}不存在---------------".format(file_path))
            # sys.exit(0)

    ex_str = ex_str[:-2] + ")"

    return eval(ex_str)


def normalize_image_shape(img_path, save_path, width, height):
    """
    处理图片大小到相同形状
    :param img_path: 图像路径
    :param save_path: 存储路径
    :param width: 图像宽度
    :param height: 图像高度
    """

    # 检查图片路径是否存在
    if not os.path.exists(img_path):
        print('-------------路径{}不存在-------------', img_path)
        sys.exit(0)

    # 判断存储的文件夹是否存在，不存在创建文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # 新建文件夹

    # 读取图片
    img = cv2.imread(img_path)

    # 原始图片的高宽和通道数
    o_height, o_width, o_channel = img.shape

    # 比目标图片缺少的像素数
    e_height = height - o_height
    e_width = width - o_width

    # 上下左右补齐的像素数
    top = int(e_height / 2)
    bottom = e_height - top
    left = int(e_width / 2)
    right = e_width - left

    # 补齐top
    top_arr = np.zeros((top, img.shape[1], o_channel))
    img = np.concatenate((top_arr, img), axis=0)
    # bottom 数组
    bottom_arr = np.zeros((bottom, img.shape[1], o_channel))
    img = np.concatenate((img, bottom_arr), axis=0)
    # left数组
    left_arr = np.zeros((img.shape[0], left, o_channel))
    img = np.concatenate((left_arr, img), axis=1)
    # right数组
    right_arr = np.zeros((img.shape[0], right, o_channel))
    img = np.concatenate((img, right_arr), axis=1)

    # 获取文件名
    file_name = os.path.split(img_path)[1]

    # 存储图片
    imageio.imwrite(os.path.join(save_path, file_name), img)

    print("------------------图像 {} 转换完成------------------".format(img_path))


def get_params(img, output_size):
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    w, h = img.size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


def data_augmentation(data_x, data_y, critical_value, send_critical_value):
    """
    处理图片大小到相同形状
    :param data_x: img图像
    :param data_y: gt图像
    :param critical_value: 进行数据增强的几率
    :param send_critical_value: 数据增强中小类发生的几率
    :return: 返回增强后的数据（Tensor）
    """

    assert data_x.shape[1] == data_y.shape[1] and data_x.shape[2] == data_y.shape[2] and len(data_x) == len(
        data_y), "img和mask大小不匹配！"
    assert (1 == data_x.shape[-1] or 3 == data_x.shape[-1] or 3 == len(data_x.shape)) and (
            1 == data_y.shape[-1] or 3 == data_y.shape[-1] or 3 == len(data_y.shape)), "数据增强方法仅能处理单通道或者多通道"

    data_x_shape = data_x.shape
    data_y_shape = data_y.shape

    # 单通道转化为Image可读取的格式
    if 4 == len(data_x.shape) and 1 == data_x.shape[-1]:
        data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], data_x.shape[2]))
    if 4 == len(data_y.shape) and 1 == data_y.shape[-1]:
        data_y = data_y.reshape((data_y.shape[0], data_y.shape[1], data_y.shape[2]))

    data_x_augmentation_rel = []
    data_y_augmentation_rel = []

    for key in range(len(data_x)):
        img = Image.fromarray(np.uint8(data_x[key]))
        mask = Image.fromarray(np.uint8(data_y[key]))

        if random.random() <= critical_value:
            # 随机水平翻转
            if random.random() > send_critical_value:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # 随机垂直翻转
            if random.random() > send_critical_value:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            # 逆时针90度
            if random.random() > send_critical_value:
                img = img.transpose(Image.ROTATE_90)
                mask = mask.transpose(Image.ROTATE_90)
            # 逆时针180度
            if random.random() > send_critical_value:
                img = img.transpose(Image.ROTATE_180)
                mask = mask.transpose(Image.ROTATE_180)
            # 逆时针270度
            if random.random() > send_critical_value:
                img = img.transpose(Image.ROTATE_270)
                mask = mask.transpose(Image.ROTATE_270)
            # 随机裁剪
            h, w = img.size
            img = ttf.pad(img, int(math.ceil(h * 1 / 8)))
            mask = ttf.pad(mask, int(math.ceil(h * 1 / 8)))
            i, j, h, w = get_params(img, (h, w))
            img = img.crop((j, i, j + w, i + h))
            mask = mask.crop((j, i, j + w, i + h))

        # 追加处理后的图片
        data_x_augmentation_rel.append(np.array(img))
        data_y_augmentation_rel.append(np.array(mask))

    data_x_augmentation_rel = torch.from_numpy(np.array(data_x_augmentation_rel).reshape(data_x_shape)).float()
    data_y_augmentation_rel = torch.from_numpy(np.array(data_y_augmentation_rel).reshape(data_y_shape))

    return data_x_augmentation_rel, data_y_augmentation_rel


def generator_random_patches(data_x, data_y, patch_size, patch_num, is_mask=False, critical_value=0.8):
    """
    将图片随机截取为块，并且带有随记遮罩功能
    :param data_x: img图像
    :param data_y: gt图像
    :param patch_size: 图块大小
    :param patch_num: 图块数目
    :param is_mask: 是否产生遮罩
    :param critical_value: 产生遮罩的几率
    :return: 返回切块
    """

    # 确保img和mask是匹配大小的
    assert data_x.shape[1] == data_y.shape[1] and data_x.shape[2] == data_y.shape[2] and len(data_x) == len(data_y)

    # 控制产生的patch数目在合理范围之内
    assert (max(data_x.shape[1], data_x.shape[2]) * patch_size) > patch_num, "patch_num过大将会导致结果过拟合！"

    patch_img_list = []
    patch_mask_list = []

    for key in range(len(data_x)):
        img = data_x[key]
        mask = data_y[key]

        # 图像左上右下均添加切块一般大小的边
        # img = np.concatenate((np.zeros((patch_size // 2, img.shape[1], img.shape[2])), img))
        # img = np.concatenate((img, np.zeros((patch_size // 2, img.shape[1], img.shape[2]))))
        # img = np.concatenate((np.zeros((img.shape[0], patch_size // 2, img.shape[2])), img), 1)
        # img = np.concatenate((img, np.zeros((img.shape[0], patch_size // 2, img.shape[2]))), 1)
        #
        # mask = np.concatenate((np.zeros((patch_size // 2, mask.shape[1], mask.shape[2])), mask))
        # mask = np.concatenate((mask, np.zeros((patch_size // 2, mask.shape[1], mask.shape[2]))))
        # mask = np.concatenate((np.zeros((mask.shape[0], patch_size // 2, mask.shape[2])), mask), 1)
        # mask = np.concatenate((mask, np.zeros((mask.shape[0], patch_size // 2, mask.shape[2]))), 1)

        for i in range(patch_num):
            # 随机生成切块的左上角坐标点
            random_w = random.randint(1, img.shape[1] - patch_size)
            random_h = random.randint(1, img.shape[0] - patch_size)

            # 截取得到图块patch
            patch_img = img[random_h:random_h + patch_size, random_w: random_w + patch_size, :]
            patch_mask = mask[random_h:random_h + patch_size, random_w: random_w + patch_size]
            #

            # 随机遮罩
            if is_mask:
                if random.random() <= critical_value:
                    # 产生随机遮罩的大小
                    random_m_h = random.randint(patch_size // 8, patch_size // 4)
                    random_m_w = random.randint(patch_size // 8, patch_size // 4)

                    # 产生随机遮罩的坐标点(左上角坐标)
                    random_patch_w = random.randint(1, patch_size - random_m_w)
                    random_patch_h = random.randint(1, patch_size - random_m_h)

                    # 对patch图进行遮罩
                    patch_img[random_patch_h:random_patch_h + random_m_h,
                    random_patch_w:random_patch_w + random_m_w] = 0
                    patch_mask[random_patch_h:random_patch_h + random_m_h,
                    random_patch_w:random_patch_w + random_m_w] = 0

            # 将截取到的图片存储
            patch_img_list.append(patch_img)
            patch_mask_list.append(patch_mask)

    return np.array(patch_img_list).astype(int), np.array(patch_mask_list).astype(int)


def generator_normal_patches(data_x, data_y, patch_size):
    """
    将图片逐个截取为块
    :param data_x: img图像
    :param data_y: gt图像
    :param patch_size: 图块大小
    :return: 返回切块
    """

    # 确保img和mask是匹配大小的
    assert data_x.shape[1] == data_y.shape[1] and data_x.shape[2] == data_y.shape[2] and len(data_x) == len(data_y)

    patch_img_list = []
    patch_mask_list = []

    for key in range(len(data_x)):
        img = data_x[key]
        mask = data_y[key]

        # 如果图像不能整切的话，不够的图像右、下添加黑边
        height_make = patch_size - (img.shape[0] % patch_size)
        width_make = patch_size - (img.shape[1] % patch_size)

        if 0 != height_make or 0 != width_make:
            img = np.concatenate((img, np.zeros((height_make, img.shape[1], img.shape[2]))), 0)
            img = np.concatenate((img, np.zeros((img.shape[0], width_make, img.shape[2]))), 1)

            mask = np.concatenate((mask, np.zeros((height_make, mask.shape[1], mask.shape[2]))), 0)
            mask = np.concatenate((mask, np.zeros((mask.shape[0], width_make, mask.shape[2]))), 1)

        height_point_list = list(range(0, img.shape[0], patch_size))
        width_point_list = list(range(0, img.shape[1], patch_size))

        for h in height_point_list:
            for w in width_point_list:
                # 截取得到图块patch
                patch_img = img[h:h + patch_size, w: w + patch_size]
                patch_mask = mask[h:h + patch_size, w: w + patch_size]
                # 将截取到的图片存储
                patch_img_list.append(patch_img)
                patch_mask_list.append(patch_mask)

    return np.array(patch_img_list).astype(int), np.array(patch_mask_list).astype(int)


def patches_to_img(patches, patch_size, shape):
    ih = shape[1] // patch_size if shape[1] % patch_size == 0 else (shape[1] // patch_size) + 1
    iw = shape[2] // patch_size if shape[2] % patch_size == 0 else (shape[1] // patch_size) + 1

    count = 1
    img_list = []
    row_list = []
    while count < patches.shape[0] + 1:
        if 0 == count % iw:
            row_list.append(np.concatenate(patches[count - iw:count], axis=1))

        if 0 == count % (ih * iw):
            img_list.append(np.concatenate(row_list, axis=0))
            row_list = []

        count = count + 1
    return np.array(img_list)[:, :shape[1], :shape[2]]


def mask_to_one_hot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)

    return semantic_map


def one_hot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)

    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


def data_split(data_x, data_y, tv_split):
    """
    数据集划分
    :param data_x: img图像（tensor）
    :param data_y: gt图像（tensor）
    :param tv_split: 分割比例（训练集比例）
    :return: 分割的结果
    """
    assert len(data_x) == len(data_y)
    # 数据集大小
    data_size = len(data_x)

    # 生成随机的数据索引
    data_indices = np.arange(0, data_size, 1)
    np.random.shuffle(data_indices)
    # 训练集索引
    train_indices = data_indices[:int(tv_split * data_size)]
    # 验证集索引
    val_indices = data_indices[int(tv_split * data_size):]
    # 训练集
    train_x = torch.index_select(data_x, 0, torch.tensor(train_indices).long())
    train_y = torch.index_select(data_y, 0, torch.tensor(train_indices).long())
    # 验证集
    val_x = torch.index_select(data_x, 0, torch.tensor(val_indices).long())
    val_y = torch.index_select(data_y, 0, torch.tensor(val_indices).long())

    return (train_x, train_y), (val_x, val_y)
