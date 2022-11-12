import os
import time
import argparse
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.utils.data as tud
import torch.backends.cudnn as cudnn
import numpy as np
import models


from torch.autograd import Variable
from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix
from PIL import Image
from utils import data_utils
from utils import sys_utils

font = {'size': 14}
font_title = {'size': 14}
font_legend = {'size': 14}

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

x = [i for i in range(15)]

test_path = r"F:\COVID\test\mask.npy"
test_imgs = np.load(test_path)
test_imgs[test_imgs > 0] = 1

pre_path_list = [
    r"F:\New Folder\0\pre.npy",
    r"F:\New Folder\4\pre.npy",
    r"F:\New Folder\3\pre.npy",
    r"F:\New Folder\1\pre.npy",
    r"F:\New Folder\2\pre.npy",
    r"F:\New Folder\5\pre.npy",
    r"F:\New Folder\6\pre.npy",
    r"F:\New Folder\7\pre.npy"]
pre_name_list = ["UNet", "Transformer before UNet", "Transformer after UNet", "Transformer as encoder", "Transformer as decoder", "Transformer in bottleneck", "Transformer in skip-connection", "TransIS-UNet"]
color_list = ["blue", "gray", "black", "cyan", "orange", "yellow", "pink", "red"]
marker_list = ["^", ".", "<", "*", "d", "X", "+", "1"]

f1_dict = {}
for key_f, pre_path in enumerate(pre_path_list):
    pre_imgs = np.load(pre_path)
    pre_imgs = pre_imgs.reshape((15, 512, 512))

    acc_list = []
    f1_list = []

    for key, y_pre in enumerate(pre_imgs):
        y_pre = np.ndarray.flatten(y_pre)
        y_true = np.ndarray.flatten(test_imgs[key])

        # 计算评测指标 混淆矩阵、灵敏度（Sen.）、特异率（Spec.）、准确率（Acc.）、精确率（Prec.）、Dice 相似系数、F1
        confusion = confusion_matrix(y_true, y_pre)
        # sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        # specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        accuracy = accuracy_score(y_true, y_pre)
        # precision = precision_score(y_true, y_pre, average='macro')
        f1 = f1_score(y_true, y_pre, labels=None, average='binary', sample_weight=None)

        acc_list.append(accuracy)
        f1_list.append(f1)

    f1_dict[pre_name_list[key_f]] = f1_list

fig, ax = plt.subplots()
plt.grid(axis="y")
for key, pre_name in enumerate(pre_name_list):
    plt.plot(x, f1_dict[pre_name], marker=marker_list[key], label=pre_name, color=color_list[key])

plt.title("F1-score for each image.", font_title)
plt.xlabel("samples", font)
plt.ylabel("F1", font)
# plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, prop=font_legend, borderaxespad=0)
plt.axis([0, 15, 0., 1.])
# fig.subplots_adjust(right=0.6)
# plt.savefig("./data/charts/single_F1_DRIVE.eps")
fig.savefig("./data/charts/single_F1.eps",dpi=1200,bbox_inches='tight')

# plt.show()
