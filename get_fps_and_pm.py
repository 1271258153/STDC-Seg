#!/usr/bin/env python3
"""计算 STDC-Seg 参数量和端到端推理 FPS。

使用方法：
    # 默认使用 STDC1-Seg、红外测试集、640x640、batch size=1
    python get_fps_and_pm.py --device cuda

    # 使用训练后的 checkpoint 测试
    python get_fps_and_pm.py --device cuda \
        --checkpoint output/infrared_images/stdc1_seg_640/pths/model_maxmIOU75.pth

    # 测试 STDC2-Seg 或启用 EMA 的模型
    python get_fps_and_pm.py --backbone STDCNet1446 --device cuda
    python get_fps_and_pm.py --use-ema --device cuda

    # 使用 CUDA FP16 测试
    python get_fps_and_pm.py --device cuda --fp16

计时范围：磁盘读图与解码、缩放、归一化、CPU 到 GPU 传输、模型 forward、
主 logits 必要时的双线性上采样、argmax，以及最终分割图传回 CPU。
不包含模型构建、checkpoint 加载和结果保存。
"""

import argparse
import os
import statistics
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.model_stages import BiSeNet


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile STDC-Seg parameters and end-to-end inference FPS')
    parser.add_argument(
        '--backbone', default='STDCNet813',
        choices=['STDCNet813', 'STDCNet1446'],
        help='STDCNet813=STDC1, STDCNet1446=STDC2')
    parser.add_argument('--num-classes', default=10, type=int,
                        help='number of segmentation classes')
    parser.add_argument('--height', default=640, type=int,
                        help='input height')
    parser.add_argument('--width', default=640, type=int,
                        help='input width')
    parser.add_argument(
        '--image-dir', default='data/infrared_images/images/test',
        help='directory containing input images (searched recursively)')
    parser.add_argument(
        '--mean', default=[0.6423561, 0.1127488, 0.4362715],
        type=float, nargs=3, metavar=('R', 'G', 'B'),
        help='RGB normalization mean')
    parser.add_argument(
        '--std', default=[0.2345256, 0.1510579, 0.1994846],
        type=float, nargs=3, metavar=('R', 'G', 'B'),
        help='RGB normalization standard deviation')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='inference batch size')
    parser.add_argument('--warmup', default=50, type=int,
                        help='warmup iterations')
    parser.add_argument('--iterations', default=200, type=int,
                        help='timed batches per repeat')
    parser.add_argument('--repeats', default=3, type=int,
                        help='number of timed repeats')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='benchmark device')
    parser.add_argument('--fp16', action='store_true',
                        help='use FP16 inference (CUDA only)')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='optional .pt/.pth checkpoint')
    parser.add_argument('--use-conv-last', action='store_true',
                        help='enable the backbone conv_last layer')
    parser.add_argument('--use-ema', action='store_true',
                        help='enable EMA attention in the backbone')
    parser.add_argument('--use-boundary-2', action='store_true')
    parser.add_argument('--use-boundary-4', action='store_true')
    boundary_8_group = parser.add_mutually_exclusive_group()
    boundary_8_group.add_argument(
        '--use-boundary-8', dest='use_boundary_8', action='store_true',
        help='return the stride-8 boundary output (default)')
    boundary_8_group.add_argument(
        '--no-use-boundary-8', dest='use_boundary_8', action='store_false',
        help='do not return the stride-8 boundary output')
    parser.set_defaults(use_boundary_8=True)
    parser.add_argument('--use-boundary-16', action='store_true')
    return parser.parse_args()


def validate_args(args):
    for name in ('num_classes', 'height', 'width', 'batch_size',
                 'iterations', 'repeats'):
        if getattr(args, name) <= 0:
            raise ValueError('--{} must be positive'.format(
                name.replace('_', '-')))
    if args.warmup < 0:
        raise ValueError('--warmup must be non-negative')
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(
            'image directory not found: {}'.format(args.image_dir))
    if any(value <= 0 for value in args.std):
        raise ValueError('--std values must be positive')


def resolve_device(requested):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return torch.device(requested)


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'checkpoint not found: {}'.format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for container_key in ('state_dict', 'model_state_dict', 'model'):
            if (container_key in checkpoint and
                    isinstance(checkpoint[container_key], dict)):
                checkpoint = checkpoint[container_key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint does not contain a state dict')

    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []
    for key, value in checkpoint.items():
        model_key = key
        for prefix in ('module.', 'model.'):
            if model_key.startswith(prefix):
                model_key = model_key[len(prefix):]
        if model_key not in model_state:
            continue
        if model_state[model_key].shape != value.shape:
            shape_mismatch.append(model_key)
            continue
        loaded[model_key] = value

    if not loaded:
        raise RuntimeError(
            'no checkpoint entries matched the model; check --backbone, '
            '--num-classes, --use-ema and --use-conv-last')
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)

    print('Checkpoint: {}'.format(checkpoint_path))
    print('Loaded state entries: {} / {}'.format(len(loaded), len(model_state)))
    missing_count = len(model_state) - len(loaded)
    if missing_count:
        print('Missing state entries: {}'.format(missing_count))
    if shape_mismatch:
        print('Shape-mismatched entries: {}'.format(shape_mismatch))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def find_image_files(image_dir):
    image_paths = []
    for directory, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(directory, filename))
    image_paths.sort()
    if not image_paths:
        raise RuntimeError(
            'no supported images found in {}'.format(image_dir))
    return image_paths


def preprocess_image(image_path, height, width, mean, std):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('failed to read image: {}'.format(image_path))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)[:, :, ::-1] / 255.0
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    return np.ascontiguousarray(image)


def inference_with_postprocess(model, input_tensor):
    outputs = model(input_tensor)
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            'expected tensor logits, got {}'.format(type(logits)))
    if logits.shape[-2:] != input_tensor.shape[-2:]:
        logits = F.interpolate(
            logits, size=input_tensor.shape[-2:], mode='bilinear',
            align_corners=True)
    return torch.argmax(logits, dim=1)


def select_batch(image_paths, batch_index, batch_size):
    start = batch_index * batch_size
    return [
        image_paths[(start + offset) % len(image_paths)]
        for offset in range(batch_size)]


def end_to_end_inference(model, batch_paths, height, width,
                         mean, std, dtype, device):
    images = [
        preprocess_image(path, height, width, mean, std)
        for path in batch_paths]
    cpu_batch = np.stack(images, axis=0)
    input_tensor = torch.from_numpy(cpu_batch).to(
        device=device, dtype=dtype, non_blocking=False)
    return inference_with_postprocess(model, input_tensor).cpu()


def benchmark(model, image_paths, args, mean, std, dtype, device):
    with torch.inference_mode():
        for batch_index in range(args.warmup):
            batch_paths = select_batch(
                image_paths, batch_index, args.batch_size)
            end_to_end_inference(
                model, batch_paths, args.height, args.width,
                mean, std, dtype, device)
        synchronize(device)

        results = []
        for repeat_index in range(args.repeats):
            synchronize(device)
            start = time.perf_counter()
            for batch_index in range(args.iterations):
                dataset_index = repeat_index * args.iterations + batch_index
                batch_paths = select_batch(
                    image_paths, dataset_index, args.batch_size)
                end_to_end_inference(
                    model, batch_paths, args.height, args.width,
                    mean, std, dtype, device)
            synchronize(device)
            elapsed = time.perf_counter() - start

            latency_ms = elapsed * 1000.0 / args.iterations
            fps = args.batch_size * args.iterations / elapsed
            results.append((latency_ms, fps))
    return results


def build_model(args):
    return BiSeNet(
        backbone=args.backbone,
        n_classes=args.num_classes,
        use_boundary_2=args.use_boundary_2,
        use_boundary_4=args.use_boundary_4,
        use_boundary_8=args.use_boundary_8,
        use_boundary_16=args.use_boundary_16,
        use_conv_last=args.use_conv_last,
        use_ema=args.use_ema)


def main():
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    if args.fp16 and device.type != 'cuda':
        raise ValueError('--fp16 is supported only with CUDA')

    torch.backends.cudnn.benchmark = device.type == 'cuda'
    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        model.half()

    image_paths = find_image_files(args.image_dir)
    mean = np.asarray(args.mean, dtype=np.float32).reshape((1, 1, 3))
    std = np.asarray(args.std, dtype=np.float32).reshape((1, 1, 3))
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)

    sample_paths = select_batch(image_paths, 0, args.batch_size)
    with torch.inference_mode():
        prediction = end_to_end_inference(
            model, sample_paths, args.height, args.width,
            mean, std, dtype, device)
    synchronize(device)

    model_name = 'STDC1-Seg' if args.backbone == 'STDCNet813' else 'STDC2-Seg'
    print('Model: {} ({})'.format(model_name, args.backbone))
    print('EMA / conv_last: {} / {}'.format(args.use_ema, args.use_conv_last))
    print('Device: {} ({})'.format(device, dtype))
    if device.type == 'cuda':
        print('GPU: {}'.format(torch.cuda.get_device_name(device)))
    print('Image directory: {}'.format(os.path.abspath(args.image_dir)))
    print('Images found: {}'.format(len(image_paths)))
    print('Input shape: {}'.format(
        (args.batch_size, 3, args.height, args.width)))
    print('Prediction shape: {}'.format(tuple(prediction.shape)))
    print('Timing scope: read/decode + resize/normalize + H2D + forward '
          '+ optional upsample/argmax + D2H')
    print('Parameters: {:,} ({:.6f} M)'.format(
        total_params, total_params / 1e6))
    print('Trainable parameters: {:,} ({:.6f} M)'.format(
        trainable_params, trainable_params / 1e6))
    print('Warmup / iterations / repeats: {} / {} / {}'.format(
        args.warmup, args.iterations, args.repeats))

    results = benchmark(
        model, image_paths, args, mean, std, dtype, device)
    latencies = [item[0] for item in results]
    fps_values = [item[1] for item in results]
    for index, (latency, fps) in enumerate(results, start=1):
        print('Repeat {}: latency={:.3f} ms/batch, FPS={:.3f} images/s'.format(
            index, latency, fps))

    print('Average latency: {:.3f} ms/batch'.format(
        statistics.mean(latencies)))
    print('Average FPS: {:.3f} images/s'.format(
        statistics.mean(fps_values)))
    if len(results) > 1:
        print('FPS std: {:.3f}'.format(statistics.stdev(fps_values)))


if __name__ == '__main__':
    main()
