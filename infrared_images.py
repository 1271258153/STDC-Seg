#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Infrared images dataset for STDC-Seg.

Reads samples from a lst file (same format as ddrnet):
    images/train/xxx.png   labels/train/xxx.png
For test/eval lst (single column), only image path is needed.

Returns (img, label) consistent with `cityscapes.py`:
    img:   float tensor [3, H, W], normalized
    label: int64 numpy  [1, H, W], values in [0, num_classes-1], 255 = ignore
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np

from transform import ColorJitter, HorizontalFlip, RandomScale, RandomCrop, Compose


class InfraredImages(Dataset):
    # 10 类标签的颜色列表 [R, G, B]
    COLOR_LIST = [
        [0, 0, 0],         # 0 _background_
        [220, 20, 60],     # 1 BL_Device
        [30, 144, 255],    # 2 CC_Server
        [50, 205, 50],     # 3 DP_Server
        [255, 165, 0],     # 4 KDVideo_Device
        [148, 0, 211],     # 5 KVM_Switcher
        [0, 206, 209],     # 6 SP_Cloud
        [255, 215, 0],     # 7 VPN_Gateway
        [255, 105, 180],   # 8 WEB_Firewall
        [128, 128, 128],   # 9 YP_Server
    ]
    CLASS_NAMES = ['_background_', 'BL_Device', 'CC_Server', 'DP_Server',
                   'KDVideo_Device', 'KVM_Switcher', 'SP_Cloud', 'VPN_Gateway',
                   'WEB_Firewall', 'YP_Server']

    def __init__(self, rootpth, list_path, cropsize=(1024, 512), mode='train',
                 randomscale=(0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
                              1.125, 1.25, 1.375, 1.5),
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 ignore_lb=255, data_subdir='infrared_images', *args, **kwargs):
        super(InfraredImages, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        self.ignore_lb = ignore_lb
        self.cropsize = cropsize
        self.data_root = osp.join(rootpth, data_subdir)

        self.imgs = []
        self.labels = []
        self.names = []
        with open(osp.join(rootpth, list_path), 'r') as fr:
            for line in fr:
                parts = line.strip().split()
                if not parts:
                    continue
                self.imgs.append(osp.join(self.data_root, parts[0]))
                # 文件名 (去扩展名), 用于保存预测结果
                self.names.append(osp.splitext(osp.basename(parts[0]))[0])
                # val/train/trainval have label; test may not
                if len(parts) > 1:
                    self.labels.append(osp.join(self.data_root, parts[1]))
                else:
                    self.labels.append(None)

        self.len = len(self.imgs)
        print('InfraredImages', self.mode, 'len', self.len)

        # pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.trans_train = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize),
        ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        img = Image.open(impth).convert('RGB')

        if self.mode in ('train', 'trainval'):
            lbpth = self.labels[idx]
            label = Image.open(lbpth)
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            img = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
            return img, label

        # val / test: resize to cropsize so batch stacking works
        # (infrared images have mixed resolutions unlike Cityscapes)
        # cropsize is (W, H)
        W, H = self.cropsize
        img = img.resize((W, H), Image.BILINEAR)
        img = self.to_tensor(img)
        if self.labels[idx] is not None:
            label = Image.open(self.labels[idx])
            label = label.resize((W, H), Image.NEAREST)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
        else:
            # no gt available (pure inference)
            label = np.full((1, H, W), self.ignore_lb, dtype=np.int64)
        return img, label

    def __len__(self):
        return self.len

    def label2color(self, label):
        """单通道 label -> RGB 彩色图, 用于可视化."""
        color_map = np.zeros(label.shape + (3,), dtype=np.uint8)
        for i, v in enumerate(self.COLOR_LIST):
            color_map[label == i] = v
        return color_map

    def save_pred(self, pred, sv_path, name):
        """保存彩色 mask. pred: [H, W] int (类别 id)."""
        if not osp.exists(sv_path):
            os.makedirs(sv_path, exist_ok=True)
        color = self.label2color(pred)
        Image.fromarray(color).save(osp.join(sv_path, name + '.png'))


if __name__ == '__main__':
    from tqdm import tqdm
    ds = InfraredImages('./data', list_path='list/infrared_images/val.lst', mode='val')
    uni = []
    for im, lb in tqdm(ds):
        uni.extend(np.unique(lb).tolist())
    print(sorted(set(uni)))
