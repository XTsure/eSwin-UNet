import sys
import numpy as np
import os  # 遍历文件夹

from utils import data_utils


def nii_files_to_png(nii_folder_path, png_folder_path):
    """
    nii文件存储为png
    :param nii_folder_path: nii文件存储根目录
    :param png_folder_path: png文件存储路径
    """

    # 检查路径是否合法
    if not os.path.exists(nii_folder_path):
        print("-----------------路径不存在-----------------")
        sys.exit(0)

    # 遍历文件夹
    dir_name_list = os.listdir(nii_folder_path)
    # 循环处理每个文件夹下的文件
    for dir_name in dir_name_list:
        # 此文件或文件夹的路径
        dir_path = os.path.join(nii_folder_path, dir_name)
        save_path = os.path.join(png_folder_path, dir_name)

        # 判断是文件还是文件夹
        if os.path.isfile(dir_path):
            # 判断是不是nii文件
            if ".nii" in dir_path:
                # png图片的路径
                png_path = save_path.replace('.nii', '')
                # nii转png
                data_utils.nii_2_png(dir_path, png_path)
        else:
            # 进入下一级文件夹
            nii_files_to_png(dir_path, save_path)


def save_img_array(img_folder_path, npy_folder_path, file_name, is_mask=False):
    """
    存储图片数组
    :param img_folder_path: 图片文件夹路径
    :param npy_folder_path: npy存储文件夹路径
    :param file_name: 文件名
    :param is_mask: 是否是灰度图
    """

    if not os.path.exists(img_folder_path):
        print("-----------------img路径错误-----------------")
        sys.exit(0)

    # png存储为npy文件
    if is_mask:
        data_utils.img_2_npy(img_folder_path, npy_folder_path, file_name, mode="GRAY")
    else:
        data_utils.img_2_npy(img_folder_path, npy_folder_path, file_name)

    print("----------------文件转换成功，存储位置为" + os.path.join(npy_folder_path, file_name) + "----------------")


def read_npy_file(file_path):
    """
    处理一个文件夹下的所有npy文件
    :param file_path: 文件夹路径
    :return: 返回数据集合
    """

    # 初始化数据存储
    npy_data = np.array([])

    # 当前文件列表
    dir_name_list = os.listdir(file_path)

    # 循环文件
    for dir_name in dir_name_list:
        # 文件路径
        path = os.path.join(file_path, dir_name)
        # 检查是不是npy文件
        if os.path.isfile(path) and ".npy" in dir_name:
            # 拼接数据
            if 0 == npy_data.size:
                npy_data = np.load(path)
            else:
                a = np.load(path)
                npy_data = np.concatenate((npy_data, np.load(path)), axis=0)
        else:
            print("--------------该文件无法处理，path：" + path + "--------------")

    return npy_data


def concat_data(npy_path, data_path, folder_list, save_folder_list):
    """
    将数据整理成 image, lung_mask, infection_mask, lung_and_infection_mask 存储
    :param npy_path: npy以及文件夹路径
    :param data_path: 存储路径
    :param folder_list: 文件夹列表
    :param save_folder_list: 目标存储的文件名
    """

    # 检查文件路径是不是存在
    if not os.path.exists(npy_path):
        print("-----------------img路径错误-----------------")
        sys.exit(0)

    # 如果文件路径不存在，建立路径
    if not os.path.exists(data_path):
        os.makedirs(data_path)  # 新建文件夹

    for key, folder in enumerate(folder_list):
        np.save(os.path.join(data_path, save_folder_list[key] + ".npy"), read_npy_file(os.path.join(npy_path, folder)))

    print("----------------------文件存储完成----------------------")


def normalize_image(folder_path, save_path, width, height):
    """
    将小于目标图片大小的所有图片加黑边
    :param folder_path: 存放图片的文件夹路径
    :param save_path: 保存路径
    :param width: 目标图片宽度
    :param height: 目标图片高度
    """
    # 获取文件夹线的所有文件名
    dir_name_list = os.listdir(folder_path)

    # 遍历文件进行处理
    for dir_name in dir_name_list:
        # 文件路径
        path = os.path.join(folder_path, dir_name)
        # 只处理png图片
        if os.path.isfile(path) and ".png" in dir_name:
            data_utils.normalize_image_shape(path, save_path, width, height)

    print("---------------转换结束---------------")


def COVID19_CT_Seg_20cases_png_2_npy(png_root, save_root):
    """
    处理20 cases CT png图像为npy 文件
    :param png_root: png文件位置
    :param save_root: npy文件存放位置
    """

    if not os.path.exists(png_root):
        print("--------------图片根目录{}不存在--------------".format(png_root))
        sys.exit(0)

    dir_name_list = os.listdir(png_root)
    file_count = 1

    for dir_name in dir_name_list:
        file_count_str = str(file_count)

        if file_count < 10:
            file_count_str = "0" + file_count_str

        if "Mask" in png_root:
            save_img_array(os.path.join(png_root, dir_name), save_root, file_count_str + '.npy',
                           is_mask=True)
        else:
            save_img_array(os.path.join(png_root, dir_name), save_root, file_count_str + '.npy')

        file_count = file_count + 1


def COVID19_CT_Seg_20cases_1_data_processing(nii_folder_path, png_folder_path, npy_folder_path, dataset_path):
    """
    处理20 cases CT图像生成npy数据集全过程
    :param nii_folder_path: nii文件存放位置
    :param png_folder_path: png文件存储位置
    :param npy_folder_path: npy文件存储位置
    :param dataset_path: 数据集文件存储位置
    """

    # nii 转 png
    nii_files_to_png(nii_folder_path, png_folder_path)  # nii_files_to_png('F:\\COVID19', 'F:\\COVID19-png')

    # 特殊大小的文件
    folder_arr = ['COVID-19-CT-Seg_20cases', 'Infection_Mask', 'Lung_and_Infection_Mask', 'Lung_Mask']

    # 循环处理每个文件夹下的png文件到npy文件
    for folder in folder_arr:
        COVID19_CT_Seg_20cases_png_2_npy(os.path.join(png_folder_path, folder), os.path.join(npy_folder_path, folder))

    # 拼接并保存数据
    concat_data(npy_folder_path, dataset_path,
                ["COVID-19-CT-Seg_20cases", "Infection_Mask", "Lung_and_Infection_Mask", "Lung_Mask"],
                ["img", "gt", "lung_and_infection_mask", "lung_mask"])


def COVID19_CT_Seg_20cases_2_data_processing(nii_folder_path, png_folder_path, npy_folder_path, dataset_path):
    """
    处理20 cases CT图像生成npy数据集全过程
    :param nii_folder_path: nii文件存放位置
    :param png_folder_path: png文件存储位置
    :param npy_folder_path: npy文件存储位置
    :param dataset_path: 数据集文件存储位置
    """

    # nii 转 png
    nii_files_to_png(nii_folder_path, png_folder_path)  # nii_files_to_png('F:\\COVID19', 'F:\\COVID19-png')

    # 特殊大小的文件
    folder_arr = ['COVID-19-CT-Seg_20cases', 'Infection_Mask', 'Lung_and_Infection_Mask', 'Lung_Mask']
    spe_img_arr = ['radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm', 'radiopaedia_14_85914_0',
                   'radiopaedia_14_85914_0', 'radiopaedia_14_85914_0']

    # 循环处理特殊大小的文件到相同大小
    for i in range(4):
        spe_png_path = os.path.join(png_folder_path, folder_arr[i], spe_img_arr[i])
        normalize_image(spe_png_path, spe_png_path, 630, 630)

    # 循环处理每个文件夹下的png文件到npy文件
    for folder in folder_arr:
        COVID19_CT_Seg_20cases_png_2_npy(os.path.join(png_folder_path, folder), os.path.join(npy_folder_path, folder))

    # 拼接并保存数据
    concat_data(npy_folder_path, dataset_path,
                ["COVID-19-CT-Seg_20cases", "Infection_Mask", "Lung_and_Infection_Mask", "Lung_Mask"],
                ["img", "gt", "lung_and_infection_mask", "lung_mask"])


def COVID_19_CT_1_data_processing(nii_folder_path, png_folder_path, npy_folder_path, dataset_path):
    # nii 转 png
    nii_files_to_png(nii_folder_path, png_folder_path)  # nii_files_to_png('F:\\COVID19', 'F:\\COVID19-png')

    folder_arr = ['img', 'mask', 'lung_mask']

    # 循环处理每个文件夹下的png文件到npy文件
    for folder in folder_arr:
        is_mask = False

        if "mask" in folder:
            is_mask = True

        save_img_array(os.path.join(png_folder_path, folder), os.path.join(npy_folder_path, folder), folder + '.npy',
                       is_mask=is_mask)

    # 拼接并保存数据
    concat_data(npy_folder_path, dataset_path, folder_arr, folder_arr)


if __name__ == '__main__':
    # 处理20 cases CT 图像 1
    # COVID19_CT_Seg_20cases_1_data_processing('F:\\COVID193', 'F:\\COVID193-png', 'F:\\COVID193-npy',
    #                                          'F:\\COVID193_data')

    # 处理20 cases CT 图像 2
    # COVID19_CT_Seg_20cases_2_data_processing('F:\\COVID192', 'F:\\COVID192-png', 'F:\\COVID192-npy', 'F:\\COVID192_data')

    COVID_19_CT_1_data_processing(r'F:\COVID\test', r'F:\COVID_PNG\test', 'F:\\COVID_NPY\\test',
                                  'F:\\COVID_DATA\\test')
