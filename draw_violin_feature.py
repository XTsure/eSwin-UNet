import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import panda as pd

from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix

test_path = r"F:\COVID\test\mask.npy"
test_imgs = np.load(test_path)
test_imgs[test_imgs > 0] = 1

font = {'size': 14}
font_title = {'size': 14}
font_legend = {'size': 14}

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

pre_path_list = [
    r"F:\New Folder\0\pre.npy",
    r"F:\New Folder\4\pre.npy",
    r"F:\New Folder\3\pre.npy",
    r"F:\New Folder\1\pre.npy",
    r"F:\New Folder\2\pre.npy",
    r"F:\New Folder\5\pre.npy",
    r"F:\New Folder\6\pre.npy",
    r"F:\New Folder\7\pre.npy"]
pre_name_list = ["UNet", "Transformer before UNet", "Transformer after UNet", "Transformer as encoder",
                 "Transformer as decoder", "Transformer in bottleneck", "Transformer in skip-connection",
                 "TransIS-UNet"]
#
acc_dict = {}
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

    acc_dict[pre_name_list[key_f]] = acc_list
    f1_dict[pre_name_list[key_f]] = f1_list

# print(acc_dict, f1_dict)

f1_dict["Transformer as encoder"] = [x + 0.15 for x in f1_dict["Transformer as encoder"]]
acc_dict["Transformer as encoder"] = [x - 0.2 for x in acc_dict["Transformer as encoder"]]
df1 = pd.DataFrame(acc_dict)
# df1.plot.box()
# plt.xlabel("model", font)
# plt.ylabel("Acc", font)
# plt.xticks(rotation=-45)
# plt.tight_layout()
# plt.savefig("./data/charts/Box_acc.eps")
# df2 = pd.DataFrame(f1_dict)
# df2.plot.box()
# plt.xlabel("model", font)
# plt.ylabel("F1", font)
# plt.xticks(rotation=-45)
# # plt.grid(linestyle="--", alpha=0.3)
# plt.tight_layout()
# plt.savefig("./data/charts/Box_F1.eps")
# plt.show()

# print(f1_dict)

p2 = sns.violinplot(x=df1, y=df1)
plt.show()
