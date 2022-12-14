import os
import time
import argparse
import torch
import torch.utils.data as tud
import torch.backends.cudnn as cudnn
import numpy as np
import visdom
import models
from torch.nn.modules.loss import CrossEntropyLoss
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix
from loss import SoftDiceLoss
from loss import DiceLoss
from utils import data_utils
from utils import sys_utils
from utils.metric_utils import dice_coeff, my_dice_coeff
from loss import MultiLoss


def get_args():
    model_names = sorted(
        name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 1)')
    parser.add_argument('--patch_num', type=int, default=2, help='input batch size for training (default: 128)')
    parser.add_argument('--patch_size', type=int, default=224, help='input batch size for training (default: 128)')
    parser.add_argument('--patch_mask_probability', type=float, default=0.8,
                        help='probability of mask in dynamic patch')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--model', type=str, default='UNet', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: FCN)')
    parser.add_argument('--data_path', type=str, default='F://COVID', help='data path')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
    parser.add_argument('--img_mode', type=str, default='RGB', choices=["gray", "RGB"], help='image color mode')
    parser.add_argument('--class_num', type=int, default=2, help='number of segmentation categories')
    parser.add_argument('--data_augmentation', action='store_false', help='whether to use data augmentation')
    parser.add_argument('--augmentation_probability', type=float, default=0.8, help='probability of data augmentation')
    parser.add_argument('--augmentation_diversity_probability', type=float, default=0.5,
                        help='probability of discrepancy in data augmentation methods')
    parser.add_argument('--img_patch', action='store_false', help='whether to use image patching technology')
    parser.add_argument('--load_last', action='store_true', help='load last model')
    parser.add_argument('--load_path', type=str, default=r'logs/UNet/20211101030430/epochs', help='load model path')
    parser.add_argument('--logs_path', type=str, default=r'logs', help='load model path')
    parser.add_argument('--tv_split', type=float, default=0.95,
                        help='division of training set and validation set(training set ratio)')

    return vars(parser.parse_args())


def train(args):
    net = args["net"]

    max_sensitivity = args["max_sensitivity"]
    max_specificity = args["max_specificity"]
    max_accuracy = args["max_accuracy"]
    max_precision = args["max_precision"]
    max_pt = args["max_pt"]
    max_f1 = args["max_f1"]

    # ?????????????????????
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

    loss_list = []

    # ????????????
    for epoch in range(args["start_epoch"], args["epochs"]):
        # ???dataset?????????tensor??????
        train_x, train_y = args["train_data"]
        if sys_args["img_patch"]:
            train_x, train_y = data_utils.generator_random_patches(data_x=train_x.numpy(), data_y=train_y.numpy(),
                                                                   patch_size=args["patch_size"],
                                                                   patch_num=args["patch_num"],
                                                                   critical_value=args["patch_mask_probability"])
        # ????????????
        if sys_args["data_augmentation"]:
            train_x, train_y = data_utils.data_augmentation(train_x, train_y, args["augmentation_probability"],
                                                            args["augmentation_diversity_probability"])
        # mask one_hot ??????(????????????????????????????????????1?????????????????????one_hot??????)
        train_y = torch.from_numpy(data_utils.mask_to_one_hot(train_y.numpy(), args["one_hot_list"]))

        # ?????????????????????
        train_data_augmentation = tud.TensorDataset(train_x, train_y)
        # ??????????????????
        train_loader = tud.DataLoader(train_data_augmentation, batch_size=args["batch_size"], shuffle=True)

        # ?????????
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
        # ????????????
        criterion = SoftDiceLoss.SoftDiceLoss(args["class_num"], activation='sigmoid', reduction='mean').cuda()
        # ??????????????????0
        train_loss = 0
        avg_train_loss = 0

        # ??????????????????
        net.train()

        # ??????????????????
        progress_bar = tqdm(train_loader, unit='patches')

        # ??????batch_size???????????????
        for step, (img, gt) in enumerate(progress_bar):
            # ?????????????????????
            progress_bar.set_description('Epoch {}-{}'.format(epoch, args["epochs"]))

            # ??????????????????????????????
            img = Variable(img.permute(0, 3, 1, 2).cuda())
            gt = Variable(gt.long().permute(0, 3, 1, 2).cuda())

            # ??????????????????????????????
            out = net(img)

            # ????????????
            loss = criterion(out, gt)  # loss of final result

            # ???????????????????????????????????????
            optimizer.zero_grad()
            # ??????????????????, ?????????????????????
            loss.backward()
            # ??????????????????????????? net ??? parameters ???
            optimizer.step()
            # ????????????
            train_loss += loss.item()
            avg_train_loss = train_loss / (step + 1)

            # ?????????????????????
            progress_bar.set_postfix(loss='%.3f' % (train_loss / (step + 1)))

        # ????????????
        val_info = val(net, args)

        loss_list.append(avg_train_loss)

        # ?????????????????????
        max_sensitivity = val_info["sensitivity"] if max_sensitivity < val_info["sensitivity"] else max_sensitivity
        max_specificity = val_info["specificity"] if max_specificity < val_info["specificity"] else max_specificity
        max_accuracy = val_info["accuracy"] if max_accuracy < val_info["accuracy"] else max_accuracy
        max_precision = val_info["precision"] if max_precision < val_info["precision"] else max_precision
        max_pt = epoch if max_f1 < val_info["f1"] else max_pt
        max_f1 = val_info["f1"] if max_f1 < val_info["f1"] else max_f1

        val_info_max = {"max_sensitivity": max_sensitivity, "max_specificity": max_specificity,
                        "max_accuracy": max_accuracy, "max_precision": max_precision, "max_f1": max_f1}

        # ????????????????????????
        val_info_str = ""
        for key, value in val_info.items():
            if "confusion" == key:
                val_info_str = val_info_str + "{}:\n{}\n".format(key, value)
            else:
                val_info_str = val_info_str + "{}    : {:.4f}    ".format(key, value)

        val_info_max_str = "\n"
        for key, value in val_info_max.items():
            val_info_max_str = val_info_max_str + "{}: {:.4f}    ".format(key, value)
        sys_utils.output_info("=", "????????????", val_info_str + val_info_max_str)
        print("max_pt", max_pt)

        # ????????????
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'args': args,
            'max_sensitivity': max_sensitivity,
            'max_specificity': max_specificity,
            'max_accuracy': max_accuracy,
            'max_precision': max_precision,
            'max_pt': max_pt,
            'max_f1': max_f1,
        }

        # ????????????
        state_save_path = args["state_path"]
        if not os.path.exists(state_save_path):
            os.makedirs(state_save_path)
        torch.save(state, os.path.join(state_save_path, "{}.pt".format(epoch)))

        # ????????????
        if 0 == epoch % 9:
            pt_list = os.listdir(state_path)
            for pt_file in pt_list:
                if pt_file != (str(max_pt) + ".pt"):
                    pt_file_path = os.path.join(state_path, pt_file)
                    print("????????????{}".format(pt_file_path))
                    os.remove(pt_file_path)

    np.save("data/loss/{}_loss.npy".format(args["model"]), np.array(loss_list))


def val(net, args):
    net.eval()
    predictions = []

    # ?????????
    val_x, val_y = args["val_data"]
    val_y_row = val_y
    img_shape = val_x.shape
    mask_shape = val_y.shape

    if sys_args["img_patch"]:
        val_x, val_y = data_utils.generator_normal_patches(val_x.numpy(), val_y.numpy(), args["patch_size"])
    val_dataset = tud.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    # val_dataset = tud.TensorDataset(val_x, val_y)
    val_loader = tud.DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    # ??????
    with torch.no_grad():
        for batch_idx, (val_in, a) in enumerate(val_loader):
            val_in = Variable(val_in.permute(0, 3, 1, 2).cuda())
            val_out = net(val_in.float())
            val_out = torch.softmax(val_out, dim=1)
            val_out = val_out.permute(0, 2, 3, 1)
            predictions.append(val_out.data.cpu().numpy())

    # ???????????????
    predictions = np.concatenate(predictions, axis=0)
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    # one_hot???????????????mask
    y_pre = data_utils.one_hot_to_mask(predictions, args["one_hot_list"]).reshape(
        (-1, args["patch_size"], args["patch_size"], mask_shape[-1]))
    # np.save("y_pre_raw.npy", y_pre)
    y_pre = data_utils.patches_to_img(y_pre, args["patch_size"], mask_shape)
    y_true = np.ndarray.flatten(val_y)
    # y_true = np.ndarray.flatten(val_y.numpy())
    y_true = np.ndarray.flatten(val_y_row.numpy())
    y_pre = np.ndarray.flatten(y_pre)

    # ?????????????????? ???????????????????????????Sen.??????????????????Spec.??????????????????Acc.??????????????????Prec.??????Dice ???????????????F1
    confusion = confusion_matrix(y_true, y_pre)
    sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    accuracy = accuracy_score(y_true, y_pre)
    precision = precision_score(y_true, y_pre, average='binary')
    f1 = f1_score(y_true, y_pre, labels=None, average='binary', sample_weight=None)

    return {"confusion": confusion, "sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy,
            "precision": precision, "f1": f1}


if __name__ == '__main__':
    # ??????????????????
    sys_args = get_args()
    sys_utils.output_info("-", "????????????", str(sys_args))

    # ?????????????????? GPU ??????
    os.environ['CUDA_VISIBLE_DEVICES'] = sys_args['device']
    # ?????? Benchmark ???????????????????????????
    cudnn.benchmark = True
    # ?????? GPU ????????????
    print("GPU{}????????????:{}".format(sys_args['device'], torch.cuda.is_available()))
    # ????????????????????????????????????????????????????????????????????????????????????
    torch.cuda.manual_seed(sys_args['seed'])

    # ???????????????
    data_x, data_y, _ = data_utils.load_data(os.path.join(sys_args["data_path"], "train"), ["img", "mask", "lung_mask"])
    # numpy ??? tensor
    data_x = torch.from_numpy(data_x).float()
    data_y = torch.from_numpy(data_y.reshape((data_y.shape[0], data_y.shape[1], data_y.shape[2], 1)))
    # ???????????????????????????????????????
    if "gray" == sys_args["img_mode"]:
        data_x = data_x[:, :, :, 0:1]

    # one_hot ?????????
    data_one_hot_list = [0, 85, 170, 255]
    data_y[data_y == 127] = 85
    if 2 == sys_args["class_num"]:
        data_y[data_y > 0] = 1
        data_one_hot_list = [0, 1]

    # ???????????????
    # train_data, val_data = data_utils.data_split(data_x, data_y, sys_args["tv_split"])
    train_data = (data_x, data_y)
    val_x, val_y, _ = data_utils.load_data(os.path.join(sys_args["data_path"], "test"), ["img", "mask", "lung_mask"])
    val_x = torch.from_numpy(val_x).float()
    val_y = torch.from_numpy(val_y.reshape((val_y.shape[0], val_y.shape[1], val_y.shape[2], 1)))
    if "gray" == sys_args["img_mode"]:
        val_x = val_x[:, :, :, 0:1]

    if 2 == sys_args["class_num"]:
        val_y[val_y > 0] = 1
    val_data = (val_x, val_y)

    # ???????????????
    img_channel = 1 if "gray" == sys_args["img_mode"] else 3

    model = models.__dict__[sys_args['model']](sys_args["patch_size"], img_channel, sys_args["class_num"]).cuda()
    # ?????????????????????
    load_path = os.path.join(sys_args["load_path"])
    # ???????????????
    parameters = sum([torch.numel(param) for param in model.parameters()]) / (1024 * 1024)
    print("?????? {} ???????????????: {:.2f} M".format(sys_args["model"], parameters))

    # ??????????????????
    state_path = os.path.join(sys_args["logs_path"], sys_args["model"],
                              time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())), "epochs")

    # ????????????
    start_epoch = 0
    max_sensitivity = 0
    max_specificity = 0
    max_accuracy = 0
    max_precision = 0
    max_pt = 0
    max_f1 = 0
    if sys_args["load_last"]:
        checkpoints = torch.load(sys_utils.get_last_model(load_path))
        model.load_state_dict(checkpoints["net"])
        start_epoch = checkpoints['epoch'] + 1
        state_path = sys_args['load_path']
        max_sensitivity = checkpoints['max_sensitivity']
        max_specificity = checkpoints['max_specificity']
        max_accuracy = checkpoints['max_accuracy']
        max_precision = checkpoints['max_precision']
        max_pt = checkpoints['max_pt']
        max_f1 = checkpoints['max_f1']

    # ????????????
    train_args = {
        "train_data": train_data,
        "val_data": val_data,
        "net": model,
        "one_hot_list": data_one_hot_list,
        "state_path": state_path,
        "start_epoch": start_epoch,
        "max_sensitivity": max_sensitivity,
        "max_specificity": max_specificity,
        "max_accuracy": max_accuracy,
        "max_precision": max_precision,
        "max_pt": max_pt,
        "max_f1": max_f1
    }

    a = {**sys_args, **train_args}
    # ????????????
    train({**sys_args, **train_args})
