#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Default config for STDC-Seg (yacs based, mirrors ddrnet's config style)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 50
_C.PIN_MEMORY = False
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Dataset related params
_C.DATASET = CN()
_C.DATASET.DATASET = 'cityscapes'        # dataset module name: cityscapes | infrared_images
_C.DATASET.ROOT = './data'
_C.DATASET.TRAIN_SET = ''                # lst path relative to ROOT, e.g. list/infrared_images/train.lst
_C.DATASET.VAL_SET = ''
_C.DATASET.TEST_SET = ''
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.IGNORE_LABEL = 255
_C.DATASET.MEAN = [0.485, 0.456, 0.406]
_C.DATASET.STD = [0.229, 0.224, 0.225]
_C.DATASET.CLASS_NAMES = []  # 类别名, 用于评估日志打印逐类 IoU/Acc; 留空则用 cls0..clsN

# Model related params
_C.MODEL = CN()
_C.MODEL.BACKBONE = 'STDCNet813'         # STDCNet813=STDC1, STDCNet1446=STDC2
_C.MODEL.PRETRAINED = ''
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.USE_CONV_LAST = False
_C.MODEL.USE_EMA = False                 # 在 backbone 深层 (feat16/feat32) 加 EMA 注意力
_C.MODEL.USE_BOUNDARY_2 = False
_C.MODEL.USE_BOUNDARY_4 = False
_C.MODEL.USE_BOUNDARY_8 = False
_C.MODEL.USE_BOUNDARY_16 = False

# Loss related params
_C.LOSS = CN()
_C.LOSS.SCORE_THRES = 0.7                # OHEM score threshold
_C.LOSS.USE_DETAIL_AGGREGATE = True       # DetailAggregateLoss (boundary)

# Training params
_C.TRAIN = CN()
_C.TRAIN.IMAGE_SIZE = [1024, 512]        # [width, height] -> cropsize = [W, H]
_C.TRAIN.BASE_SIZE = 1024
_C.TRAIN.RANDOM_SCALE = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
                         1.125, 1.25, 1.375, 1.5]
_C.TRAIN.BATCH_SIZE_PER_GPU = 8          # n_img_per_gpu
_C.TRAIN.N_WORKERS_TRAIN = 4
_C.TRAIN.N_WORKERS_VAL = 1
_C.TRAIN.SHUFFLE = False                 # DistributedSampler handles shuffling
_C.TRAIN.MAX_ITER = 40000
_C.TRAIN.SAVE_ITER_SEP = 1000
_C.TRAIN.WARMUP_STEPS = 1000
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.LR = 0.01                       # lr_start
_C.TRAIN.WD = 0.0005
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.POWER = 0.9                     # poly lr decay exponent
_C.TRAIN.WARMUP_START_LR = 0.00001
_C.TRAIN.RESPATH = 'output/stdc_seg'     # where checkpoints/logs are saved

# Evaluation params
_C.EVAL = CN()
_C.EVAL.BATCH_SIZE = 2
_C.EVAL.SCALE_50 = 0.5                    # corresponds to model_maxmIOU50.pth
_C.EVAL.SCALE_75 = 0.75                  # corresponds to model_maxmIOU75.pth
_C.EVAL.MODEL_FILE = ''


def update_config(cfg, args):
    cfg.defrost()
    if getattr(args, 'cfg', None):
        cfg.merge_from_file(args.cfg)
    if getattr(args, 'opts', None):
        cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
