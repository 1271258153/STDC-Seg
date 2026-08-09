#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""生成横向四联对比图: 原图 | 真值 | 预测 | 叠加

用法:
    python comparison.py
    python comparison.py --cfg experiments/infrared_images/stdc1_seg.yaml
    python comparison.py --cfg <yaml> EVAL.MODEL_FILE <path_to_pth>

输出:
    output/infrared_images/comparison_images/<name>.png
"""

from logger import setup_logger
from models.model_stages import BiSeNet
from infrared_images import InfraredImages
from config import config, update_config

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import os.path as osp
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("请先安装 matplotlib: pip install matplotlib")


def build_eval_dataset(cfg, cropsize):
    list_path = cfg.DATASET.TEST_SET if cfg.DATASET.TEST_SET else cfg.DATASET.VAL_SET
    ds = InfraredImages(cfg.DATASET.ROOT, list_path=list_path,
                        cropsize=cropsize, mode='val',
                        mean=tuple(cfg.DATASET.MEAN),
                        std=tuple(cfg.DATASET.STD),
                        ignore_lb=cfg.DATASET.IGNORE_LABEL)
    return ds, list_path


def denormalize(img_t, mean, std):
    """[3,H,W] normalized tensor -> [H,W,3] uint8 (RGB)"""
    mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    img = img_t.cpu().numpy() * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img.transpose(1, 2, 0)


def overlay(img_rgb, color_mask, alpha=0.5):
    """原图与彩色 mask 叠加. img_rgb/color_mask: [H,W,3] uint8"""
    img_f = img_rgb.astype(np.float32)
    mask_f = color_mask.astype(np.float32)
    out = (1 - alpha) * img_f + alpha * mask_f
    return np.clip(out, 0, 255).astype(np.uint8)


def make_comparison(cfg):
    logger = logging.getLogger()
    logger.info('=' * 20)
    logger.info('generating comparison images ...')
    logger.info(config)

    cropsize = list(cfg.TRAIN.IMAGE_SIZE)  # [W, H]
    dsval, list_path = build_eval_dataset(cfg, cropsize)
    dl = DataLoader(dsval, batch_size=1, shuffle=False,
                    num_workers=cfg.WORKERS, drop_last=False)

    n_classes = cfg.DATASET.NUM_CLASSES
    net = BiSeNet(backbone=cfg.MODEL.BACKBONE, n_classes=n_classes,
                  use_boundary_2=cfg.MODEL.USE_BOUNDARY_2,
                  use_boundary_4=cfg.MODEL.USE_BOUNDARY_4,
                  use_boundary_8=cfg.MODEL.USE_BOUNDARY_8,
                  use_boundary_16=cfg.MODEL.USE_BOUNDARY_16,
                  use_conv_last=cfg.MODEL.USE_CONV_LAST,
                  use_ema=cfg.MODEL.USE_EMA)
    net.load_state_dict(torch.load(cfg.EVAL.MODEL_FILE), strict=False)
    net.cuda()
    net.eval()

    sv_dir = osp.join(cfg.OUTPUT_DIR, cfg.DATASET.DATASET, 'comparison_images')
    os.makedirs(sv_dir, exist_ok=True)
    logger.info('saving comparison images to: %s', sv_dir)

    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD
    scale = cfg.EVAL.SCALE_75
    ignore_label = cfg.DATASET.IGNORE_LABEL

    with torch.no_grad():
        for idx, (imgs, label) in enumerate(tqdm(dl, desc='comparison')):
            name = dsval.names[idx]
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]  # [H, W] (cropsize)

            imgs = imgs.cuda()
            N, C, H, W = imgs.size()
            new_hw = [int(H * scale), int(W * scale)]
            in_imgs = F.interpolate(imgs, new_hw, mode='bilinear',
                                    align_corners=True) if scale != 1.0 else imgs

            logits = net(in_imgs)[0]
            logits = F.interpolate(logits, size=size, mode='bilinear',
                                    align_corners=True)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            # 原图: 直接读原始文件, 保持原图尺寸/色彩
            orig_img = Image.open(dsval.imgs[idx]).convert('RGB')
            orig_w, orig_h = orig_img.size
            img_rgb = np.array(orig_img)  # [H, W, 3] uint8

            # 把 label / pred resize 回原图尺寸
            lb_np = label[0].cpu().numpy().astype(np.uint8)
            if (orig_h, orig_w) != size:
                lb_np = np.array(Image.fromarray(lb_np).resize(
                    (orig_w, orig_h), Image.NEAREST)).astype(np.uint8)
                pred = np.array(Image.fromarray(pred).resize(
                    (orig_w, orig_h), Image.NEAREST)).astype(np.uint8)

            lb_color = dsval.label2color(lb_np)
            pred_color = dsval.label2color(pred)
            overlaid = overlay(img_rgb, pred_color, alpha=0.5)

            # 拼成 1x4 横向连接
            fig, axes = plt.subplots(1, 4, figsize=(48, 12))
            titles = ['Image', 'Ground Truth', 'Prediction', 'Overlay']
            images = [img_rgb, lb_color, pred_color, overlaid]
            for ax, im, t in zip(axes, images, titles):
                ax.imshow(im)
                ax.set_title(t, fontsize=20)
                ax.axis('off')
            plt.tight_layout()
            out_path = osp.join(sv_dir, name + '.png')
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)

    logger.info('done. %d comparison images saved to %s', len(dsval), sv_dir)


def parse_args():
    parse = argparse.ArgumentParser(description='Generate 1x4 horizontal comparison images')
    parse.add_argument('--cfg', dest='cfg', type=str,
                       default='experiments/infrared_images/stdc1_seg.yaml',
                       help='experiment configure file name')
    parse.add_argument('opts', help='Modify config options using the command-line',
                       default=None, nargs=argparse.REMAINDER)
    args = parse.parse_args()
    update_config(config, args)
    return args


if __name__ == '__main__':
    args = parse_args()
    log_dir = config.TRAIN.RESPATH if config.TRAIN.RESPATH else config.OUTPUT_DIR
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    setup_logger(log_dir)
    make_comparison(config)
