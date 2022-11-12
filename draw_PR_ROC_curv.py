import os
import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from utils import data_utils

data_path = '/home/sdc_3_7T/jiangyun/wuchao/dataset/COVID/100CT/test'  # 数据集路径

font = {'size': 14}
font_title = {'size': 16}
font_legend = {'size': 12}

test_gts = np.load(os.path.join(data_path, 'mask.npy'))
test_gts[test_gts > 0] = 1
test_gts = test_gts.reshape((15, 512, 512, 1))
img_shape = test_gts.shape
data_one_hot_list = [0, 1]
test_gts = data_utils.mask_to_one_hot(test_gts, data_one_hot_list)
# test_gts = test_gts / 255.
# test_masks = load_hdf5(data_path + 'border_masks_test.hdf5')

model_1 = "/home/sdc_3_7T/jiangyun/wuchao/COVID19/UNetMRFG/20210504132031/epochs/30/predictions/pre.npy"
model_1_pred_imgs = np.load(model_1)
model_1_pred_imgs = model_1_pred_imgs.reshape(img_shape)
model_1_pred_imgs = data_utils.mask_to_one_hot(model_1_pred_imgs, data_one_hot_list)

model_2 = "/home/sdc_3_7T/jiangyun/wuchao/COVID19/UNet/20210426002805/epochs/331/predictions/pre.npy"
model_2_pred_imgs = np.load(model_2)
model_2_pred_imgs = model_2_pred_imgs.reshape(img_shape)
model_2_pred_imgs = data_utils.mask_to_one_hot(model_2_pred_imgs, data_one_hot_list)

model_3 = "/home/sdc_3_7T/jiangyun/wuchao/COVID19/UNetMRFG/20210504132040/epochs/125/predictions/pre.npy"
model_3_pred_imgs = np.load(model_3)
model_3_pred_imgs = model_3_pred_imgs.reshape(img_shape)
model_3_pred_imgs = data_utils.mask_to_one_hot(model_3_pred_imgs, data_one_hot_list)

model_4 = "/home/sdc_3_7T/jiangyun/wuchao/COVID19/Vit_Aggregation_UNet/20210503023647/epochs/315/predictions/pre.npy"
model_4_pred_imgs = np.load(model_4)
model_4_pred_imgs = model_4_pred_imgs.reshape(img_shape)
model_4_pred_imgs = data_utils.mask_to_one_hot(model_4_pred_imgs, data_one_hot_list)

model_5 = "/home/sdc_3_7T/jiangyun/wuchao/COVID19/Vit_Aggregation_UNet/20210504132800/epochs/9/predictions/pre.npy"
model_5_pred_imgs = np.load(model_5)
model_5_pred_imgs = model_5_pred_imgs.reshape(img_shape)
model_5_pred_imgs = data_utils.mask_to_one_hot(model_5_pred_imgs, data_one_hot_list)

model_6 = "/home/sdc_3_7T/jiangyun/wuchao/COVID19/Vit_In_BottleNeck/20210507015344/epochs/231/predictions/pre.npy"
model_6_pred_imgs = np.load(model_6)
model_6_pred_imgs = model_6_pred_imgs.reshape(img_shape)
model_6_pred_imgs = data_utils.mask_to_one_hot(model_6_pred_imgs, data_one_hot_list)

# AUC_ROC
plt.figure(1)

y_scores_1 = np.asarray(model_1_pred_imgs).astype(np.float32).flatten()
y_true = np.asarray(test_gts).astype(np.float32).flatten()
fpr_1, tpr_1, thresholds_1 = roc_curve((y_true), y_scores_1)
AUC_ROC_1 = roc_auc_score(y_true, y_scores_1)

y_scores_2 = np.asarray(model_2_pred_imgs).astype(np.float32).flatten()
fpr_2, tpr_2, thresholds_2 = roc_curve((y_true), y_scores_2)
AUC_ROC_2 = roc_auc_score(y_true, y_scores_2)

y_scores_3 = np.asarray(model_3_pred_imgs).astype(np.float32).flatten()
fpr_3, tpr_3, thresholds_3 = roc_curve((y_true), y_scores_3)
AUC_ROC_3 = roc_auc_score(y_true, y_scores_3)

y_scores_4 = np.asarray(model_4_pred_imgs).astype(np.float32).flatten()
fpr_4, tpr_4, thresholds_4 = roc_curve((y_true), y_scores_4)
AUC_ROC_4 = roc_auc_score(y_true, y_scores_4)

y_scores_5 = np.asarray(model_5_pred_imgs).astype(np.float32).flatten()
fpr_5, tpr_5, thresholds_5 = roc_curve((y_true), y_scores_5)
AUC_ROC_5 = roc_auc_score(y_true, y_scores_5)

y_scores_6 = np.asarray(model_6_pred_imgs).astype(np.float32).flatten()
fpr_6, tpr_6, thresholds_6 = roc_curve((y_true), y_scores_6)
AUC_ROC_6 = roc_auc_score(y_true, y_scores_6)

plt.plot(fpr_1, tpr_1, label='TBU (AUC = %0.4f)' % AUC_ROC_1, color='black')
plt.plot(fpr_2, tpr_2, label='TIS (AUC = %0.4f)' % AUC_ROC_2, color='yellow')
plt.plot(fpr_3, tpr_3, label='TIS-UNet (AUC = %0.4f)' % AUC_ROC_3, color='red')
plt.plot(fpr_4, tpr_4, label='T-A-U (AUC = %0.4f)' % AUC_ROC_4, color='green')
plt.plot(fpr_5, tpr_5, label='UNet (AUC = %0.4f)' % AUC_ROC_5, color='blue')
plt.plot(fpr_6, tpr_6, label='T-I-B (AUC = %0.4f)' % AUC_ROC_6, color='orange')

plt.xlabel("FPR (False Positive Rate)", font)
plt.ylabel("TPR (True Positive Rate)", font)
plt.legend(loc="lower right", prop=font_legend)
plt.axis([0, 1, 0.90, 1])
plt.title('ROC curve', font_title)
plt.savefig("./data/charts/AUC_ROC.eps")
plt.show()

# Precision Recall (PR) curve
plt.figure(2)

precision_1, recall_1, thresholds_1 = precision_recall_curve(y_true, y_scores_1)
AUC_prec_rec_1 = auc(precision_1, recall_1)

precision_2, recall_2, thresholds_2 = precision_recall_curve(y_true, y_scores_2)
AUC_prec_rec_2 = auc(precision_2, recall_2)

precision_3, recall_3, thresholds_3 = precision_recall_curve(y_true, y_scores_3)
AUC_prec_rec_3 = auc(precision_3, recall_3)

precision_4, recall_4, thresholds_4 = precision_recall_curve(y_true, y_scores_4)
AUC_prec_rec_4 = auc(precision_4, recall_4)

precision_5, recall_5, thresholds_5 = precision_recall_curve(y_true, y_scores_5)
AUC_prec_rec_5 = auc(precision_5, recall_5)

precision_6, recall_6, thresholds_6 = precision_recall_curve(y_true, y_scores_6)
AUC_prec_rec_6 = auc(precision_6, recall_6)

plt.plot(recall_1, precision_1, '-', label='TBU (AUC = %0.4f)' % AUC_prec_rec_1, color='black')
plt.plot(recall_2, precision_2, '-', label='TIS (AUC = %0.4f)' % AUC_prec_rec_2, color='yellow')
plt.plot(recall_3, precision_3, '-', label='TIS-UNet (AUC = %0.4f)' % AUC_prec_rec_3, color='red')
plt.plot(recall_4, precision_4, '-', label='T-A-U (AUC = %0.4f)' % AUC_prec_rec_4, color='green')
plt.plot(recall_5, precision_5, '-', label='UNet (AUC = %0.4f)' % AUC_prec_rec_5, color='blue')
plt.plot(recall_6, precision_6, '-', label='T-I-B (AUC = %0.4f)' % AUC_prec_rec_6, color='blue')

plt.title('Precision Recall curve', font_title)
plt.xlabel("Recall", font)
plt.ylabel("Precision", font)
plt.legend(loc="lower left", prop=font_legend)
plt.axis([0, 1, 0.90, 1])
plt.savefig("./data/charts/AUC_PR.eps")
plt.show()
