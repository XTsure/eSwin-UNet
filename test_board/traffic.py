import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 生成 hopf 1239 张
hopf1 = np.random.rand(5, 1) * 0.1  # 0-0.1
hopf2 = np.random.rand(16, 1) * 0.1 + 0.1  # 0.1-0.2
hopf3 = np.random.rand(39, 1) * 0.1 + 0.2  # 0.2-0.3
hopf4 = np.random.rand(37, 1) * 0.1 + 0.3  # 0.3-0.4
hopf5 = np.random.rand(68, 1) * 0.1 + 0.4  # 0.4-0.5
hopf6 = np.random.rand(132, 1) * 0.1 + 0.5  # 0.5-0.6
hopf7 = np.random.rand(407, 1) * 0.1 + 0.6  # 0.6-0.7
hopf8 = np.random.rand(381, 1) * 0.1 + 0.7  # 0.7-0.8
hopf9 = np.random.rand(123, 1) * 0.1 + 0.8  # 0.8-0.9
hopf10 = np.random.rand(31, 1) * 0.1 + 0.9  # 0.9-1

hopf1_d1 = np.random.rand(5, 1) * 0.1 + 0.8  # 0-0.9
hopf2_d1 = np.random.rand(16, 1) * 0.1 + 0.7  # 0-0.8
hopf3_d1 = np.random.rand(39, 1) * 0.1 + 0.6  # 0-0.7
hopf4_d1 = np.random.rand(37, 1) * 0.1 + 0.5  # 0-0.6
hopf5_d1 = np.random.rand(68, 1) * 0.1 + 0.4  # 0.4-0.5
hopf6_d1 = np.random.rand(132, 1) * 0.1 + 0.3  # 0.5-0.4
hopf7_d1 = np.random.rand(407, 1) * 0.1 + 0.2  # 0.6-0.3
hopf8_d1 = np.random.rand(381, 1) * 0.1 + 0.1  # 0.7-0.2
hopf9_d1 = np.random.rand(123, 1) * 0.1  # 0.8-0.1
hopf10_d1 = np.random.rand(31, 1) * 0  # 0

hopf = np.concatenate((hopf1, hopf2, hopf3, hopf4, hopf5, hopf6, hopf7, hopf8, hopf9, hopf10), axis=0)
hopf_d1 = np.concatenate(
    (hopf1_d1, hopf2_d1, hopf3_d1, hopf4_d1, hopf5_d1, hopf6_d1, hopf7_d1, hopf8_d1, hopf9_d1, hopf10_d1), axis=0)

# 生成  前堵1766
crowd_fh1 = np.random.rand(18, 1) * 0.1  # 0-0.1
crowd_fh2 = np.random.rand(39, 1) * 0.1 + 0.1  # 0.1-0.2
crowd_fh3 = np.random.rand(30, 1) * 0.1 + 0.2  # 0.2-0.3
crowd_fh4 = np.random.rand(107, 1) * 0.1 + 0.3  # 0.3-0.4
crowd_fh5 = np.random.rand(71, 1) * 0.1 + 0.4  # 0.4-0.5
crowd_fh6 = np.random.rand(256, 1) * 0.1 + 0.5  # 0.5-0.6
crowd_fh7 = np.random.rand(570, 1) * 0.1 + 0.6  # 0.6-0.7
crowd_fh8 = np.random.rand(494, 1) * 0.1 + 0.7  # 0.7-0.8
crowd_fh9 = np.random.rand(172, 1) * 0.1 + 0.8  # 0.8-0.9
crowd_fh10 = np.random.rand(9, 1) * 0.1 + 0.9  # 0.9-1

crowd_fh1_d1 = np.random.rand(18, 1) * 0.1 + 0.8  # 0-0.9
crowd_fh2_d1 = np.random.rand(39, 1) * 0.1 + 0.7  # 0-0.8
crowd_fh3_d1 = np.random.rand(30, 1) * 0.1 + 0.6  # 0-0.7
crowd_fh4_d1 = np.random.rand(107, 1) * 0.1 + 0.5  # 0-0.6
crowd_fh5_d1 = np.random.rand(71, 1) * 0.1 + 0.4  # 0.4-0.5
crowd_fh6_d1 = np.random.rand(256, 1) * 0.1 + 0.3  # 0.5-0.4
crowd_fh7_d1 = np.random.rand(570, 1) * 0.1 + 0.2  # 0.6-0.3
crowd_fh8_d1 = np.random.rand(494, 1) * 0.1 + 0.1  # 0.7-0.2
crowd_fh9_d1 = np.random.rand(172, 1) * 0.1  # 0.8-0.1
crowd_fh10_d1 = np.random.rand(9, 1) * 0  # 0

crowd_fh = np.concatenate(
    (crowd_fh1, crowd_fh2, crowd_fh3, crowd_fh4, crowd_fh5, crowd_fh6, crowd_fh7, crowd_fh8, crowd_fh9, crowd_fh10),
    axis=0)

crowd_fh_d1 = np.concatenate(
    (crowd_fh1_d1, crowd_fh2_d1, crowd_fh3_d1, crowd_fh4_d1, crowd_fh5_d1, crowd_fh6_d1, crowd_fh7_d1, crowd_fh8_d1,
     crowd_fh9_d1, crowd_fh10_d1), axis=0)

# 生成 后堵1907张
crowd_lh1 = np.random.rand(32, 1) * 0.1  # 0-0.1
crowd_lh2 = np.random.rand(21, 1) * 0.1 + 0.1  # 0.1-0.2
crowd_lh3 = np.random.rand(67, 1) * 0.1 + 0.2  # 0.2-0.3
crowd_lh4 = np.random.rand(62, 1) * 0.1 + 0.3  # 0.3-0.4
crowd_lh5 = np.random.rand(115, 1) * 0.1 + 0.4  # 0.4-0.5
crowd_lh6 = np.random.rand(491, 1) * 0.1 + 0.5  # 0.5-0.6
crowd_lh7 = np.random.rand(317, 1) * 0.1 + 0.6  # 0.6-0.7
crowd_lh8 = np.random.rand(663, 1) * 0.1 + 0.7  # 0.7-0.8
crowd_lh9 = np.random.rand(117, 1) * 0.1 + 0.8  # 0.8-0.9
crowd_lh10 = np.random.rand(22, 1) * 0.1 + 0.9  # 0.9-1

crowd_lh1_d1 = np.random.rand(32, 1) * 0.1 + 0.8  # 0-0.1
crowd_lh2_d1 = np.random.rand(21, 1) * 0.1 + 0.7  # 0.1-0.2
crowd_lh3_d1 = np.random.rand(67, 1) * 0.1 + 0.6  # 0.2-0.3
crowd_lh4_d1 = np.random.rand(62, 1) * 0.1 + 0.5  # 0.3-0.4
crowd_lh5_d1 = np.random.rand(115, 1) * 0.1 + 0.4  # 0.4-0.5
crowd_lh6_d1 = np.random.rand(491, 1) * 0.1 + 0.3  # 0.5-0.6
crowd_lh7_d1 = np.random.rand(317, 1) * 0.1 + 0.2  # 0.6-0.7
crowd_lh8_d1 = np.random.rand(663, 1) * 0.1 + 0.1  # 0.7-0.8
crowd_lh9_d1 = np.random.rand(117, 1) * 0.1  # 0.8-0.9
crowd_lh10_d1 = np.random.rand(22, 1) * 0  # 0.9-1

crowd_lh = np.concatenate(
    (crowd_lh1, crowd_lh2, crowd_lh3, crowd_lh4, crowd_lh5, crowd_lh6, crowd_lh7, crowd_lh8, crowd_lh9, crowd_lh10),
    axis=0)

crowd_lh_d1 = np.concatenate(
    (crowd_lh1_d1, crowd_lh2_d1, crowd_lh3_d1, crowd_lh4_d1, crowd_lh5_d1, crowd_lh6_d1, crowd_lh7_d1, crowd_lh8_d1,
     crowd_lh9_d1, crowd_lh10_d1), axis=0)

hopf_d2 = 1 - hopf - hopf_d1
crowd_fh_d2 = 1 - crowd_fh - crowd_fh_d1
crowd_lh_d2 = 1 - crowd_lh - crowd_lh_d1

hopf_r = np.concatenate((hopf, hopf_d1, hopf_d2), axis=1)
crowd_fh_r = np.concatenate((crowd_fh_d1, crowd_fh, crowd_fh_d2), axis=1)
crowd_lh_r = np.concatenate((crowd_lh_d1, crowd_lh_d2, crowd_lh), axis=1)

hopf_label = np.zeros((1239, 1))
crowd_fh_label = np.ones((1766, 1))
crowd_lh_label = np.ones((1907, 1)) * 2

traffic_label = np.concatenate((hopf_label, crowd_fh_label, crowd_lh_label), axis=0)
traffic_pre = np.concatenate((hopf_r, crowd_fh_r, crowd_lh_r), axis=0)

traffic = np.concatenate((traffic_pre, traffic_label), axis=1)

np.random.shuffle(traffic)

traffic_pre = traffic[:, 0:3]
traffic_label = traffic[:, 3]

np.save("traffic_pre.npy", traffic_pre)
np.save("traffic_label.npy", traffic_label)
