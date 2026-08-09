#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from models.model_stages import BiSeNet
from cityscapes import CityScapes
from infrared_images import InfraredImages
from config import config, update_config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import math
import argparse

class MscEvalV0(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:

            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            logits = net(imgs)[0]
  
            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes).float()
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()

def evaluatev0(respth='./pretrained', dspth='./data', backbone='CatNetSmall', scale=0.75, use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)
    ## dataset
    batchsize = 5
    n_workers = 2
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    n_classes = 19
    print("backbone:", backbone)
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
     use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
     use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
     use_conv_last=use_conv_last)
    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()
    

    with torch.no_grad():
        single_scale = MscEvalV0(scale=scale)
        mIOU, ious, accs = single_scale(net, dl, 19)
    logger = logging.getLogger()
    logger.info('mIOU is: %s\n', mIOU)

class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75],
            n_classes = 19,
            lb_ignore = 255,
            cropsize = 1024,
            flip = True,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model


    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0]-H, size[1]-W
        hst, hed = margin_h//2, margin_h//2+H
        wst, wed = margin_w//2, margin_w//2+W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5/6.
        N, C, H, W = im.size()
        long_size, short_size = (H,W) if H>W else (W,H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        else:
            stride = math.ceil(cropsize*stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W-cropsize)/stride)+1
            n_y = math.ceil((H-cropsize)/stride)+1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(H, stride*iy+cropsize), min(W, stride*ix+cropsize)
                    hst, wst = hed-cropsize, wed-cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        return prob


    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H*scale), int(W*scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb==ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


    def evaluate(self):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank()==0:
            dloader = self.dl
        for i, (imgs, label) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                # prob = self.scale_crop_eval(imgs, sc)
                prob = self.eval_chip(imgs)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate(respth='./resv1_catnet/pths/', dspth='./data'):
    ## logger
    logger = logging.getLogger()

    ## model
    logger.info('\n')
    logger.info('===='*20)
    logger.info('evaluating the model ...\n')
    logger.info('setup and restore model')
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)

    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    ## dataset
    batchsize = 5
    n_workers = 2
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(net, dl, scales=[1], flip = False)

    ## eval
    mIOU = evaluator.evaluate()
    logger.info('mIOU is: {:.6f}'.format(mIOU))



def build_eval_dataset(cfg, cropsize):
    name = cfg.DATASET.DATASET
    list_path = cfg.DATASET.TEST_SET if cfg.DATASET.TEST_SET else cfg.DATASET.VAL_SET
    if name == 'cityscapes':
        return CityScapes(cfg.DATASET.ROOT, mode='val'), list_path
    elif name == 'infrared_images':
        ds = InfraredImages(cfg.DATASET.ROOT, list_path=list_path,
                            cropsize=cropsize, mode='val',
                            mean=tuple(cfg.DATASET.MEAN),
                            std=tuple(cfg.DATASET.STD),
                            ignore_lb=cfg.DATASET.IGNORE_LABEL)
        return ds, list_path
    else:
        raise ValueError('Unsupported dataset: {}'.format(name))


def evaluate_cfg(cfg):
    """Config-driven evaluation entry.

    - 默认读 DATASET.TEST_SET (evaluation.lst)
    - 打印 MeanIU / Pixel_Acc / Mean_Acc / Class IoU
    - 保存彩色 mask 到 output/<dataset>/evaluation_result/
    """
    logger = logging.getLogger()
    logger.info('===='*20)
    logger.info('evaluating the model ...')
    logger.info(config)

    cropsize = list(cfg.TRAIN.IMAGE_SIZE)  # [W, H]
    dsval, list_path = build_eval_dataset(cfg, cropsize)
    dl = DataLoader(dsval,
                    batch_size = 1,                 # 保存 mask 需逐张处理
                    shuffle = False,
                    num_workers = cfg.WORKERS,
                    drop_last = False)

    n_classes = cfg.DATASET.NUM_CLASSES
    ignore_label = cfg.DATASET.IGNORE_LABEL
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

    # 彩色 mask 保存目录: output/<dataset>/<set>_result
    set_name = osp.splitext(osp.basename(list_path))[0]
    sv_dir = osp.join(cfg.OUTPUT_DIR, cfg.DATASET.DATASET, set_name + '_result')
    logger.info('saving colored masks to: %s', sv_dir)

    # 类别名
    class_names = list(cfg.DATASET.CLASS_NAMES) if len(cfg.DATASET.CLASS_NAMES) > 0 \
        else ['cls{}'.format(i) for i in range(n_classes)]

    hist = torch.zeros(n_classes, n_classes).cuda().detach()
    scale = cfg.EVAL.SCALE_75
    with torch.no_grad():
        for idx, (imgs, label) in enumerate(tqdm(dl)):
            name = dsval.names[idx]
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()
            N, C, H, W = imgs.size()
            new_hw = [int(H*scale), int(W*scale)]
            in_imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True) \
                if scale != 1.0 else imgs

            logits = net(in_imgs)[0]
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
            preds = torch.argmax(logits, dim=1)  # [N, H, W]

            # 累加混淆矩阵
            keep = label != ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
            ).view(n_classes, n_classes).float()

            # 保存彩色 mask (resize 回原图尺寸)
            pred_np = preds[0].cpu().numpy().astype(np.uint8)
            if hasattr(dsval, 'save_pred'):
                orig_w, orig_h = Image.open(dsval.imgs[idx]).size
                if (orig_h, orig_w) != size:
                    pred_pil = Image.fromarray(pred_np).resize(
                        (orig_w, orig_h), Image.NEAREST)
                    pred_np = np.array(pred_pil).astype(np.uint8)
                dsval.save_pred(pred_np, sv_dir, name)

            if idx % 100 == 0:
                pos = hist.sum(1); res = hist.sum(0); tp = hist.diag()
                iou_arr = tp / torch.clamp(pos + res - tp, min=1.0)
                logger.info('processing: %d images, running mIoU: %.4f',
                             idx, float(iou_arr.mean()))

    # 最终指标
    pos = hist.sum(1); res = hist.sum(0); tp = hist.diag()
    pixel_acc = float(tp.sum() / torch.clamp(pos.sum(), min=1.0))
    mean_acc = float((tp / torch.clamp(pos, min=1.0)).mean())
    iou_arr = tp / torch.clamp(pos + res - tp, min=1.0)
    mean_iou = float(iou_arr.mean())

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, Mean_Acc: {: 4.4f}, Class IoU: '.format(
        mean_iou, pixel_acc, mean_acc)
    logger.info(msg)
    iou_list = ['{:.4f}'.format(float(x)) for x in iou_arr.cpu().numpy()]
    logger.info('  '.join(iou_list))
    # 逐类带名字
    logger.info('per-class:')
    for ci, cname in enumerate(class_names):
        logger.info('  {:3d} {:<15s} IoU={:.4f} Acc={:.4f}'.format(
            ci, cname, float(iou_arr[ci]),
            float(tp[ci] / torch.clamp(pos[ci], min=1.0))))


def parse_eval_args():
    parse = argparse.ArgumentParser(description='Evaluate STDC-Seg')
    parse.add_argument('--cfg', dest='cfg', type=str,
                       default='experiments/infrared_images/stdc1_seg.yaml',
                       help='experiment configure file name')
    parse.add_argument('opts', help='Modify config options using the command-line',
                       default=None, nargs=argparse.REMAINDER)
    args = parse.parse_args()
    update_config(config, args)
    return args


if __name__ == "__main__":
    args = parse_eval_args()
    # 日志放到 TRAIN.RESPATH (如 output/infrared_images/stdc1_seg_640/)
    log_dir = config.TRAIN.RESPATH if config.TRAIN.RESPATH else config.OUTPUT_DIR
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    setup_logger(log_dir)
    evaluate_cfg(config)

