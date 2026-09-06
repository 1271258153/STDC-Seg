## 训练自己的数据集
1. 将图片、标签和list文件放入 `data/` 下
2. 预训练权重放入 `checkpoints/` 下
3. 修改 `experiments/infrared_images/stdc1_seg.yaml` 中的参数

### 训练
```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 train.py --cfg experiments/infrared_images/stdc1_seg.yaml && /usr/bin/shutdown
```

### 评估
```bash
python evaluation.py
```

### 生成对比图
```bash
python comparison.py
```
> 在 `output/infrared_images/comparison_images` 下生成四格对比图

### 获取参数量和FPS
```bash
python get_fps_and_pm.py --device cuda
```