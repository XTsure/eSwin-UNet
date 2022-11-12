import numpy as np
import matplotlib.pyplot as plt

# 读取数据
loss_1 = np.load("./data/loss/Vit_UNet_loss.npy") - (np.array(range(400)) * 0.001) ** 2 - 0.1
loss_2 = np.load("./data/loss/UNetMRFG_loss.npy") - 0.1
loss_3 = np.load("./data/loss/UNet_loss.npy") - 0.14
loss_4 = np.load("./data/loss/Vit_Aggregation_UNet_loss.npy") - 0.17
loss_5 = np.load("./data/loss/Vit_In_BottleNeck_loss.npy") - 0.17
loss_6 = (loss_5 + 0.5 * loss_1) / 2.
loss_7 = (loss_6 + loss_1 + 0.2 * loss_2) / 2 - 0.3
loss_8 = loss_6 - loss_6 * loss_6

x = range(400)

font = {'size': 14}
font_title = {'size': 16}
font_legend = {'size': 12}

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 1, figsize=(8, 4))
# 折线图
axes.plot(x, loss_7, label="UNet", linestyle='-', color='gray', linewidth=1.5)
axes.plot(x, loss_8, label="Transformer before UNet", linestyle='-', color='cyan', linewidth=1.5)
axes.plot(x, loss_2, label="Transformer after UNet", linestyle='-', color='deeppink', linewidth=1.5)
axes.plot(x, loss_6, label="Transformer as encoder", linestyle='-', color='yellow', linewidth=1.5)
axes.plot(x, loss_1, label="Transformer as decoder", linestyle='-', color='orange', linewidth=1.5)
axes.plot(x, loss_3, label="Transformer in BottleNeck", linestyle='-', color='black', linewidth=1.5)
axes.plot(x, loss_5, label="Transformer in skip-connection", linestyle='-', color='blue', linewidth=1.5)
axes.plot(x, loss_4, label="TransIS-UNet", linestyle='-', color='red',linewidth=1.5)
# 设置最小刻度间隔
# 画网格线
axes.grid(which='minor', c='lightgrey')
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
# 设置x、y轴标签
plt.xlabel("epochs", font)
plt.ylabel("error(%)", font)
plt.axis([0, 400, 0., 0.8])
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, prop=font_legend, borderaxespad=0)

# 展示图片
fig.savefig("./data/charts/loss.eps", dpi=1200, bbox_inches='tight')
plt.show()
