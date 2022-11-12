import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

y_true = np.load("traffic_label.npy").astype(np.int8)
traffic_pre = np.load("traffic_pre.npy")
traffic_pre_list = []
for index, item in enumerate(traffic_pre):
    traffic_pre_list.append(np.where(item == np.max(item))[0][0])

y_pred = np.array(traffic_pre_list)

np.save("y_pred.npy", y_pred)
np.save("y_true.npy", y_true)
acc = accuracy_score(y_true, y_pred)

# 条形图的绘制
plt.rcParams['font.sans-serif'] = ['SimHei']


sample_index = np.random.randint(0, y_pred.shape[-1], 6)
sample_index = np.sort(sample_index)
labels = ['p' + str(x) for x in sample_index]

width = 0.35  # 条形图的宽度
fig, ax = plt.subplots()
ax.bar(labels, traffic_pre[[sample_index]][:, 0], width, label='Hopf')
ax.bar(labels, traffic_pre[[sample_index]][:, 1], width, bottom=traffic_pre[[sample_index]][:, 0], label='saddle knot')
ax.bar(labels, traffic_pre[[sample_index]][:, 2], width,
       bottom=(traffic_pre[[sample_index]][:, 0] + traffic_pre[[sample_index]][:, 1]), label='free flow')

index = np.arange(6)
for x, y, z in zip(index-0.165, traffic_pre[[sample_index]][:, 0]/2 , traffic_pre[[sample_index]][:, 0]):
    if not round(z, 2) == 0:
        ax.text(x, y, '{:.2f}'.format(z))

for x, y, z in zip(index-0.165, traffic_pre[[sample_index]][:, 0] + traffic_pre[[sample_index]][:, 1]/2, traffic_pre[[sample_index]][:, 1]):
    if not round(z, 2) == 0:
        ax.text(x, y, '{:.2f}'.format(z))

for x, y, z in zip(index-0.165, traffic_pre[[sample_index]][:, 0] + traffic_pre[[sample_index]][:, 1] + traffic_pre[[sample_index]][:, 2]/2, traffic_pre[[sample_index]][:, 2]):
    if not round(z, 2) == 0:
        ax.text(x, y, '{:.2f}'.format(z))

ax.set_ylabel('predicted percentage')
ax.set_xlabel('number of test image')
ax.set_title('forecast result graph')
ax.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

# 展示图片
fig.savefig("traffic.png", dpi=1200, bbox_inches='tight')

plt.show()

print("--------------------------")
print("acc:{:.4f}%".format(acc*100))
print("--------------------------")