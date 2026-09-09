#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""生成预测可视化图。

用法:
    python comparison.py
    python comparison.py --cfg experiments/infrared_images/stdc1_seg.yaml
    python comparison.py --cfg <yaml> EVAL.MODEL_FILE <path_to_pth>
    python comparison.py --overlay-only
    python comparison.py --input-dir path/to/images --output-dir path/to/results
    python comparison.py --input-dir path/to/images --image example.png
    python comparison.py --overlay-only --image example.png
    python comparison.py --overlay-only --images example1.png,example2.png

``--image`` / ``--images`` 可重复使用，也可传入逗号分隔的文件名。
``--input-dir`` 会读取目录中的常见图片文件并自动只生成叠加图。
``--output-dir`` 用于直接指定结果保存目录。
文件名可以带路径或扩展名，实际会按数据集中的文件名（不含扩展名）匹配。

输出:
    output/infrared_images/comparison_images/<name>.png
"""

from logger import setup_logger
from models.model_stages import BiSeNet
from infrared_images import InfraredImages
from config import config, update_config

import torch
from torch.utils.data import DataLoader, Dataset, Subset
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


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class ImageDirectoryDataset(Dataset):
    """从指定目录读取无标注图片，用于生成预测叠加图。"""

    def __init__(self, input_dir, cropsize, mean, std, ignore_label):
        input_dir = osp.abspath(osp.expanduser(input_dir))
        if not osp.isdir(input_dir):
            raise ValueError('输入目录不存在或不是目录: {}'.format(input_dir))

        entries = sorted(os.scandir(input_dir), key=lambda entry: entry.name)
        self.imgs = [
            entry.path for entry in entries
            if entry.is_file() and osp.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not self.imgs:
            raise ValueError('输入目录中没有支持的图片: {}'.format(input_dir))

        self.names = [osp.splitext(osp.basename(path))[0] for path in self.imgs]
        self.cropsize = cropsize
        self.ignore_label = ignore_label
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        width, height = self.cropsize
        img = img.resize((width, height), Image.BILINEAR)
        img = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(img)
        img = (img - self.mean) / self.std
        label = np.full((1, height, width), self.ignore_label, dtype=np.int64)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,), dtype=np.uint8)
        for class_id, color in enumerate(InfraredImages.COLOR_LIST):
            color_map[label == class_id] = color
        return color_map

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


def normalize_image_names(images):
    """把命令行中的图片路径/文件名转成不带扩展名的文件名。"""
    if not images:
        return []

    names = []
    for value in images:
        for item in value.split(','):
            item = item.strip()
            if item:
                names.append(osp.splitext(osp.basename(item))[0])
    # 去重但保留用户指定的顺序
    return list(dict.fromkeys(names))


def select_image_indices(dataset, images):
    """返回指定图片在数据集中的下标；未指定时返回全部下标。"""
    requested = normalize_image_names(images)
    if not requested:
        return list(range(len(dataset)))

    indices_by_name = {}
    for idx, name in enumerate(dataset.names):
        indices_by_name.setdefault(name, []).append(idx)

    missing = [name for name in requested if name not in indices_by_name]
    if missing:
        examples = ', '.join(dataset.names[:10])
        raise ValueError(
            '在输入数据中找不到图片: {}. 可用名称示例: {}'.format(
                ', '.join(missing), examples or '<empty dataset>'))

    return [idx for name in requested for idx in indices_by_name[name]]


def make_comparison(cfg, overlay_only=False, images=None, input_dir=None, output_dir=None):
    logger = logging.getLogger()
    logger.info('=' * 20)
    logger.info('generating comparison images ...')
    logger.info(config)

    cropsize = list(cfg.TRAIN.IMAGE_SIZE)  # [W, H]
    if input_dir:
        dsval = ImageDirectoryDataset(
            input_dir, cropsize,
            tuple(cfg.DATASET.MEAN),
            tuple(cfg.DATASET.STD),
            cfg.DATASET.IGNORE_LABEL)
        if not overlay_only:
            logger.info('--input-dir has no labels; enabling overlay-only mode')
        overlay_only = True
    else:
        dsval, _ = build_eval_dataset(cfg, cropsize)
    selected_indices = select_image_indices(dsval, images)
    selected_ds = Subset(dsval, selected_indices)
    dl = DataLoader(selected_ds, batch_size=1, shuffle=False,
                    num_workers=cfg.WORKERS, drop_last=False)
    if images:
        logger.info('selected %d image(s): %s', len(selected_indices),
                    ', '.join(dsval.names[idx] for idx in selected_indices))

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

    sv_dir = (osp.abspath(osp.expanduser(output_dir)) if output_dir else
              osp.join(cfg.OUTPUT_DIR, cfg.DATASET.DATASET, 'comparison_images'))
    os.makedirs(sv_dir, exist_ok=True)
    logger.info('saving comparison images to: %s', sv_dir)

    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD
    scale = cfg.EVAL.SCALE_75
    ignore_label = cfg.DATASET.IGNORE_LABEL

    with torch.no_grad():
        for batch_idx, (imgs, label) in enumerate(tqdm(dl, desc='comparison')):
            idx = selected_indices[batch_idx]
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

            lb_color = None
            if not overlay_only:
                lb_color = dsval.label2color(lb_np)
            pred_color = dsval.label2color(pred)
            overlaid = overlay(img_rgb, pred_color, alpha=0.5)

            out_path = osp.join(sv_dir, name + '.png')
            if overlay_only:
                # 保存原始分辨率的纯叠加图，不添加标题或留白。
                Image.fromarray(overlaid).save(out_path)
            else:
                # 拼成 1x4 横向连接
                fig, axes = plt.subplots(1, 4, figsize=(48, 12))
                titles = ['Image', 'Ground Truth', 'Prediction', 'Overlay']
                comparison_images = [img_rgb, lb_color, pred_color, overlaid]
                for ax, im, title in zip(axes, comparison_images, titles):
                    ax.imshow(im)
                    ax.set_title(title, fontsize=20)
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(out_path, dpi=120, bbox_inches='tight')
                plt.close(fig)

    image_type = 'overlay' if overlay_only else 'comparison'
    logger.info('done. %d %s image(s) saved to %s',
                len(selected_indices), image_type, sv_dir)


def parse_args():
    parse = argparse.ArgumentParser(description='Generate prediction visualization images')
    parse.add_argument('--cfg', dest='cfg', type=str,
                       default='experiments/infrared_images/stdc1_seg.yaml',
                       help='experiment configure file name')
    parse.add_argument('--overlay-only', action='store_true',
                       help='save only the prediction overlay at the original resolution')
    parse.add_argument('--input-dir', type=str,
                       help='read input images from this directory')
    parse.add_argument('--output-dir', type=str,
                       help='save generated images directly to this directory')
    parse.add_argument('--image', '--images', dest='images', action='append',
                       metavar='IMAGE',
                       help=('only process this image; repeat the option or use a '
                             'comma-separated list for multiple images'))
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
    make_comparison(config,
                    overlay_only=args.overlay_only,
                    images=args.images,
                    input_dir=args.input_dir,
                    output_dir=args.output_dir)
