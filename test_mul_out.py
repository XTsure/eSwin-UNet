import os
import time
import argparse
import torch
import torch.utils.data as tud
import torch.backends.cudnn as cudnn
import numpy as np
import models

from torch.autograd import Variable
from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix, jaccard_score
from PIL import Image
from utils import data_utils
from utils import sys_utils


def get_args():
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 1)')
    parser.add_argument('--load_path', type=str, default=r'F:\COVID19\E_Swin_UNet\20211105230845\epochs\168.pt',
                        help='load model path')
    parser.add_argument('--default_data', action='store_true')
    parser.add_argument('--data_path', type=str, default='F://COVID', help='data path')

    return vars(parser.parse_args())


def test(args):
    save_path, file_name = os.path.split(args["load_path"])
    file_n = file_name.split('.')[0]
    pre_path = os.path.join(save_path, file_n, "predictions")
    if os.path.exists(pre_path):
        print("------------------测试结果已经存在！路径：{}------------------".format(pre_path))
    else:
        net = args["net"]
        # 使用测试模式
        net.eval().cuda()

        predictions = []

        # 验证集
        test_x, test_y, test_mask = args["test_data"]
        ori_test_y = test_y
        mask_shape = test_y.shape

        if args["img_patch"]:
            test_x, test_y = data_utils.generator_normal_patches(test_x.numpy(), test_y.numpy(), args["patch_size"])
        test_dataset = tud.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        test_loader = tud.DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

        # 预测
        start = time.time()
        with torch.no_grad():
            for batch_idx, (test_in, _) in enumerate(test_loader):
                test_in = Variable(test_in.permute(0, 3, 1, 2).cuda())
                test_out = net(test_in.float())
                test_out = test_out[1]
                test_out = torch.softmax(test_out, dim=1)
                test_out = test_out.permute(0, 2, 3, 1)
                predictions.append(test_out.data.cpu().numpy())
        end = time.time()
        # 运行时间
        operation_time = end - start
        # 二值化处理
        predictions = np.concatenate(predictions, axis=0)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0
        # one_hot编码转变为mask
        y_pre = data_utils.one_hot_to_mask(predictions, args["one_hot_list"]).reshape(
            (-1, args["patch_size"], args["patch_size"], mask_shape[-1]))
        y_pre = data_utils.patches_to_img(y_pre, args["patch_size"], mask_shape)
        y_pre = np.ndarray.flatten(y_pre)
        # 分割结果加入lung_mask
        mask = np.ndarray.flatten(test_mask.numpy())
        y_pre[mask == 0] = 0
        y_true = np.ndarray.flatten(ori_test_y.numpy())

        # 计算评测指标 混淆矩阵、灵敏度（Sen.）、特异率（Spec.）、准确率（Acc.）、精确率（Prec.）、Dice 相似系数F1
        confusion = confusion_matrix(y_true, y_pre)
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        accuracy = accuracy_score(y_true, y_pre)
        precision = precision_score(y_true, y_pre, average='binary')
        f1 = f1_score(y_true, y_pre, labels=None, average='binary', sample_weight=None)
        jaccard = jaccard_score(y_true=y_true, y_pred=y_pre, average='binary')

        test_info = {"confusion": confusion,
                     "sensitivity": sensitivity,
                     "specificity": specificity,
                     "accuracy": accuracy,
                     "precision": precision,
                     "f1": f1,
                     "jaccard": jaccard,
                     "operation_time": operation_time}

        sys_utils.output_info("-", "测试结果", test_info)

        # 存储指标
        print(pre_path)
        sys_utils.write_file(pre_path, "metrics.txt", str(test_info))
        np.save(os.path.join(pre_path, "pre.npy"), y_pre)

        # 生成图片
        y_pre = y_pre.reshape((ori_test_y.shape[0], ori_test_y.shape[1], ori_test_y.shape[2]))
        for i in range(len(y_pre)):
            im = Image.fromarray(y_pre[i] * 255)
            im = im.convert("L")
            im.save(os.path.join(pre_path, str(i) + ".png"), quality=95)

        y_true = y_true.reshape((ori_test_y.shape[0], ori_test_y.shape[1], ori_test_y.shape[2]))
        for i in range(len(y_true)):
            im = Image.fromarray(y_true[i] * 255)
            im = im.convert("L")
            im.save(os.path.join(pre_path, str(i) + "t.png"), quality=95)


if __name__ == "__main__":
    # 获取参数
    sys_args = get_args()

    # 声明要使用的 GPU 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = sys_args['device']
    # 使用 Benchmark 模式，提升计算速度
    cudnn.benchmark = True
    # 检查 GPU 是否可用
    print("GPU{}是否可用:{}".format(sys_args['device'], torch.cuda.is_available()))
    # 设置固定生成随机数的种子，使得每次运行时生成的随机数相同

    checkpoint = torch.load(sys_args["load_path"])
    train_args = checkpoint['args']

    # 加载数据
    train_args["data_path"] = train_args["data_path"] if sys_args["default_data"] else sys_args["data_path"]
    print(type(sys_args["default_data"]), sys_args["data_path"], train_args["data_path"])
    data_x, data_y, data_mask = data_utils.load_data(os.path.join(train_args["data_path"], "test"),
                                                     ["img", "mask", "lung_mask"])
    # numpy 转 tensor
    data_x = torch.from_numpy(data_x).float()
    data_y = torch.from_numpy(data_y.reshape((data_y.shape[0], data_y.shape[1], data_y.shape[2], 1)))
    data_mask = torch.from_numpy(data_mask)
    # 如果是灰度模式，使用单通道
    if "gray" == train_args["img_mode"]:
        data_x = data_x[:, :, :, 0:1]

    # one_hot 编码值
    data_one_hot_list = [0, 85, 127, 170, 255]
    if 2 == train_args["class_num"]:
        data_y[data_y > 0] = 1
        data_one_hot_list = [0, 1]

    # 加载模型
    img_channel = 1 if "gray" == train_args["img_mode"] else 3
    model = models.__dict__[train_args['model']](train_args["patch_size"], img_channel, train_args["class_num"]).cuda()
    model.load_state_dict(checkpoint['net'], strict=False)
    # 训练参数
    test_args = {
        "batch_size": sys_args["batch_size"],
        "test_data": (data_x, data_y, data_mask),
        "net": model,
        "one_hot_list": data_one_hot_list,
        "load_path": sys_args["load_path"]
    }

    # 进行测试
    test({**train_args, **test_args})
