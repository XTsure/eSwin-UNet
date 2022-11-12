import os
import numpy as np

from PIL import Image


def output_info(div_char, title, info):
    print(div_char * 50)
    print(title)
    print(info)
    print(div_char * 50)


def get_last_model(folder_path):
    """
    处理图片大小到相同形状
    :param folder_path: 文件夹路径
    :return: 返回最近修改的文件路径
    """

    assert os.path.exists(folder_path) and os.path.isdir(folder_path), "文件夹不存在！"

    # 文件列表
    dir_list = os.listdir(folder_path)
    # 文件排序
    dir_list.sort(key=lambda x: int(x[:-3]))
    # 最近修改的文件路径
    file_path = os.path.join(folder_path, dir_list[-1])

    return file_path


def show_img_from_array(img_array, filename):
    img = Image.fromarray(np.uint8(img_array))
    img.show()
    # img.save(filename)


def write_file(file_path, file_name, content):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_url = os.path.join(file_path, file_name)
    if os.path.exists(file_url):
        print("=====文件{}已经存在=====".format(file_path))
    else:
        f = open(file_url, 'w')
        f.write(content)
        f.close()
