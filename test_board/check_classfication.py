import sys
import os
import math
import random
import torch
import imageio
import cv2
import numpy as np
import nibabel as nib

from PIL import Image
from torchvision.transforms import functional as ttf
from utils import sys_utils

from utils import data_utils


img_folder_path = r'F:\COVID_PNG\train\mask'
for filename in os.listdir(img_folder_path):
    # 图片路径
    img_path = os.path.join(img_folder_path, filename)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    flag = 0
    a = np.unique(img)
    # for i in img:
    #     for j in i:
    #         if j == 170:
    #             flag = 1
    #             break
    #     if flag == 1:
    #         break
    # if flag == 1:
    print(filename, a)

        # array_of_img.append(img)