# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from .swin_unet import SwinTransformerSys
from .unet_mul import UNetMul
from .auto_encoder import AutoEncoder


class E_Swin_UNet(nn.Module):
    def __init__(self, img_size=224, in_channel=3, num_classes=2, zero_head=False, vis=False):
        super(E_Swin_UNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.auto_encoder = AutoEncoder(in_channel)

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                            # patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            # in_chans=config.MODEL.SWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=96,
                                            depths=[2, 2, 2, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=7,
                                            # mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            # qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            # qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            # drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=0.2,
                                            # ape=config.MODEL.SWIN.APE,
                                            # patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            # use_checkpoint=config.TRAIN.USE_CHECKPOINT
                                            )

        self.unet = UNetMul(in_channel, self.num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        out0 = self.auto_encoder(x)

        swin_unet_out = self.swin_unet(x)
        unet_out = self.unet(x)

        out1 = 0
        for i in range(len(unet_out)):
            out1 = out1 + swin_unet_out[i] + unet_out[i]

        out2 = swin_unet_out[-1]

        return out0, out1, out2, swin_unet_out, unet_out
