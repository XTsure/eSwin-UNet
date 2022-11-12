import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, jaccard_score

test_path = r"F:\COVID\test\mask.npy"
test_imgs = np.load(test_path)
test_imgs[test_imgs > 0] = 1

font = {'size': 14}
font_title = {'size': 14}
font_legend = {'size': 14}

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

pre_path_list = [
    r"F:\COVID19\UNet\20211107175646\epochs\128\predictions\pre.npy",
    r"F:\COVID19\AttU_Net\20211108170530\epochs\103\predictions\pre.npy",
    r"F:\COVID19\DenseUnet\20211109033121\epochs\104\predictions\pre.npy",
    r"F:\COVID19\DenseUnet\20211109033121\epochs\110\predictions\pre.npy",
    r"F:\COVID19\E_Swin_UNet\20211105230845\epochs\168\predictions\pre.npy"]
pre_name_list = ["UNet", "Swin-UNet", "UNet-MS", "Swin-UNet-MS", "eSwin-UNet"]

jaccard_dict = {}
f1_dict = {}
for key_f, pre_path in enumerate(pre_path_list):
    pre_imgs = np.load(pre_path)
    pre_imgs = pre_imgs.reshape((15, 512, 512))

    jaccard_list = []
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
        jaccard = jaccard_score(y_true=y_true, y_pred=y_pre, average='binary')

        jaccard_list.append(accuracy)
        f1_list.append(f1)

    jaccard_dict[pre_name_list[key_f]] = jaccard_list
    f1_dict[pre_name_list[key_f]] = f1_list

plt.figure()
df1 = pd.DataFrame(jaccard_dict)
f1 = df1.boxplot(sym='r*', patch_artist=True, return_type='dict', widths=0.2)
for box in f1['boxes']:
    # 箱体边框颜色
    box.set(color='#999999')
    # 箱体内部填充颜色
    box.set(facecolor='#0681d0')
for whisker in f1['whiskers']:
    whisker.set(color='lightskyblue')
for cap in f1['caps']:
    cap.set(color='lightsteelblue')
for median in f1['medians']:
    median.set(color='lightgray')
for flier in f1['fliers']:
    flier.set(marker='o', color='green', alpha=0.5)
plt.grid(alpha=0.4, linestyle='-.')
plt.xlabel("model", font)
plt.ylabel("Jaccard", font)
plt.title("")
plt.savefig("./data/charts/jaccard_F1.eps")

plt.figure()
df2 = pd.DataFrame(f1_dict)
f2 = df2.boxplot(sym='r*', patch_artist=True, return_type='dict', widths=0.2)
for box in f2['boxes']:
    # 箱体边框颜色
    box.set(color='#999999')
    # 箱体内部填充颜色
    box.set(facecolor='#0681d0')
for whisker in f2['whiskers']:
    whisker.set(color='lightskyblue')
for cap in f2['caps']:
    cap.set(color='lightsteelblue')
for median in f2['medians']:
    median.set(color='lightgray')
for flier in f2['fliers']:
    flier.set(marker='o', color='g', alpha=0.5)
plt.grid(alpha=0.4, linestyle='-.')
plt.xlabel("model", font)
plt.ylabel("Dice", font)
plt.savefig("./data/charts/Box_Dice.eps")
plt.show()
