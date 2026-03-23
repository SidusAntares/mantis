import argparse

from tqdm import tqdm


import torch.optim
import torch.nn.functional as F

from torchvision import transforms
from dataset import PixelSetData
from torch.utils.data.sampler import WeightedRandomSampler
from timematch_utils import label_utils
from collections import Counter
from dataset import PixelSetData, create_evaluation_loaders
from timematch_utils.train_utils import bool_flag
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    AddPixelLabels
)
from torch.utils import data
from sklearn.metrics import f1_score
import numpy as np
from pathlib import Path
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random

from datetime import datetime
import os
from mantis.architecture import MantisV2
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

import torch
def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models.')
    parser.add_argument('--device', default=None, type=str,
                        help='Device to use (e.g., cuda:0, cpu). Auto-detected if not specified.')
    parser.add_argument('-n', '--per', default=0.3, type=float,
                        help='percentage of labeled samples (training and validation) (default )')
    parser.add_argument('--seed', default=111, type=int,
                        help='seed')
    # 以下都是timematch
    parser.add_argument('--gpus', type=int, default=4,
                        help='Number of GPUs to use (0=CPU, 1=Single GPU, >=2=Multi-GPU DDP)')
    parser.add_argument(
        "--num_workers", default=2, type=int, help="Number of workers"
    )
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--balance_source", type=bool_flag, default=True, help='class balanced batches for source')
    parser.add_argument('--num_pixels', default=2, type=int, help='Number of pixels to sample from the input sample')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of time steps to sample from the input sample')
    # 数据路径与域
    # parser.add_argument('--data_root', default='/data/user/DBL/timematch_data', type=str,
    #                     help='Path to datasets root directory')
    parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str,
                        help='Path to datasets root directory')
    parser.add_argument('--source', default='france/31TCJ/2017', type=str)
    # parser.add_argument('--target', default='france/31TCJ/2017', type=str) denmark/32VNH/2017 austria/33UVP/2017 france/30TXT/2017
    # 类别处理
    parser.add_argument('--combine_spring_and_winter', action='store_true')
    # 数据划分
    parser.add_argument('--num_folds', default=1, type=int)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    # 评估
    parser.add_argument('--sample_pixels_val', action='store_true')  # 布尔型开关参数（flag），它不需要传值，只需在命令行中出现或不出现该选项

    args = parser.parse_args()


    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
# ================== put a patch ===============
    args.workers = args.num_workers
# ==============================================

    return args

class TimeMatchToUSCropsAdapter:
    """
    将 Batch Dict 转换为模型输入 (X, y)。
    核心功能：处理全无效时间步样本，防止模型内除零错误 (NaN)。
    """

    def __init__(self, device):
        self.device = device
        self.warn_count = 0

    def __call__(self, batch_dict):
        pixels = batch_dict['pixels']  # [B, T, C, N]
        positions = batch_dict['positions']  # [B, N] (DOY)
        valid_pixels = batch_dict['valid_pixels']  # [B, N, T] (0/1)
        pixel_labels = batch_dict['pixel_labels']  # [B, N]

        B, T, C, N = pixels.shape

        # 1. 展平维度: (B, N) -> Sample_Batch
        data_flat = pixels.permute(0, 3, 1, 2).reshape(-1, T, C)  # [S, T, C]
        doy_flat = positions.unsqueeze(1).expand(-1, N, -1).reshape(-1, T)  # [S, T]
        valid_flat = valid_pixels.permute(0, 2, 1).reshape(-1, T).bool()  # [S, T]
        mask_flat = ~valid_flat  # [S, T] (True=Invalid)
        y_flat = pixel_labels.reshape(-1).long()  # [S]

        if C > 1:
            # 假设 Sentinel-2: Red=B4=index2, NIR=B8=index6
            if C >= 7:
                red = data_flat[:, :, 2:3]  # [S, T, 1]
                nir = data_flat[:, :, 6:7]  # [S, T, 1]
                denom = nir + red + 1e-8
                ndvi = (nir - red) / denom
                data_flat = ndvi  # [S, T, 1]
            else:
                # 如果不是标准 Sentinel-2，可取均值或报错
                print("⚠️ Warning: Multi-channel input but not enough bands for NDVI. Using mean.")
                data_flat = data_flat.mean(dim=-1, keepdim=True)  # [S, T, 1]

        # 2. 【关键修复】处理全无效时间步样本 (All Invalid Time Steps)
        # 现象：随机采样可能导致某像素所有时间步都被云遮挡 (valid_flat 全 False)
        # 后果：STNet 计算权重和时分母为 0 -> NaN
        # 策略：强制将第一个时间步标记为有效，并在 Loss 中忽略该样本
        has_valid_time = valid_flat.any(dim=1)
        all_invalid_mask = ~has_valid_time

        if all_invalid_mask.any():
            count = all_invalid_mask.sum().item()
            if self.warn_count < 3:
                print(f"⚠️ Fixing {count} all-invalid samples (forcing t=0 valid).")
                self.warn_count += 1

            # 强制修正 Mask
            valid_flat[all_invalid_mask, 0] = True
            mask_flat[all_invalid_mask, 0] = False
            has_valid_time = valid_flat.any(dim=1)  # 更新状态

        # 3. 处理数据中的 NaN/Inf (数值清洗)
        # 将脏数据替换为 0，并将对应 Label 设为 Ignore (-100)
        dirty_mask = torch.isnan(data_flat) | torch.isinf(data_flat)
        if dirty_mask.any():
            sample_dirty = dirty_mask.any(dim=(1, 2))
            y_flat = y_flat.float()
            y_flat[sample_dirty] = -100.0
            y_flat = y_flat.long()
            data_flat[dirty_mask] = 0.0

            if self.warn_count < 3:
                print(f"⚠️ Cleaned {sample_dirty.sum()} samples with NaN/Inf values.")
                self.warn_count += 1

        # 4. 应用 Label Ignore
        # 如果样本依然没有有效时间步 (理论上已被步骤 2 修复，此处为双重保险)，设为 -100
        IGNORE_INDEX = -100
        y_flat = torch.where(has_valid_time, y_flat, torch.tensor(IGNORE_INDEX, dtype=y_flat.dtype))

        # 5. 最终安全检查
        data_flat = torch.nan_to_num(data_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # 构建输入元组
        X_tuple = (data_flat, mask_flat, doy_flat, valid_flat.float())

        # 6. 移至设备
        if self.device:
            X_tuple = tuple(t.to(self.device) if torch.is_tensor(t) else t for t in X_tuple)
            y_flat = y_flat.to(self.device)

        return X_tuple, y_flat

def get_data_loaders(splits, config, balance_source=True):

    strong_aug = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            Normalize(),
            ToTensor(),
            AddPixelLabels()
    ])

    source_dataset = PixelSetData(config.data_root, config.source,
            config.classes, strong_aug,
            indices=splits[config.source]['train'],)

    if balance_source:
        source_labels = source_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')

    return source_loader

def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if type(num_indices) == dict:
                indices = list(range(num_indices[dataset]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) + len(test_indices) == n

            splits[dataset] = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        folds.append(splits)
    return folds

def train(args):
    print("=> creating dataloader")
    config = cfg = args
    source_classes = label_utils.get_classes(cfg.source.split('/')[0],
                                             combine_spring_and_winter=cfg.combine_spring_and_winter)
    source_data = PixelSetData(cfg.data_root, cfg.source, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    cfg.classes = source_classes
    cfg.num_classes = len(source_classes)  # 可以覆盖该参数的默认设置

    # 控制微调样本量
    total_num = len(source_data)  # 获取全量长度
    if  args.per ==1 :
        use_num = total_num
        print(f"Using all {total_num} samples.")
    elif args.per>1 or args.per <0:
        raise ValueError('Percentage must be between 0 and 1')
    else:
        use_num = round(args.per * total_num)
        print(f"⚠️ Limiting experiment pool to {use_num} random samples (Seed={args.seed}).")
    print(f"(Seed={args.seed}).")

    # Randomly assign parcels to train/val/test
    indices = {config.source: use_num}
    folds = create_train_val_test_folds([config.source], config.num_folds, indices, config.val_ratio,
                                        config.test_ratio)
    splits = folds[0]
    sample_pixels_val = config.sample_pixels_val
    val_loader, test_loader = create_evaluation_loaders(config.source, splits, config, sample_pixels_val)
    source_loader = get_data_loaders(splits, config, config.balance_source)

    num_classes = cfg.num_classes
    args.nclasses = cfg.num_classes
    print("==========>number of classes", num_classes)


    device = torch.device(args.device)



    match args.source:
        case 'france/30TXT/2017':
            source_name = 'FR1'
        case 'france/31TCJ/2017':
            source_name = 'FR2'
        case 'denmark/32VNH/2017':
            source_name = 'DK1'
        case _:
            source_name = 'AT1'

    file_name = f'{source_name}/finetune_R{use_num}_{timestamp}_Seed{args.seed}'

    # 2. 加载预训练 Mantis 模型
    print("=> Loading pre-trained MantisV2 model...")
    network = MantisV2(device=args.device)
    network = network.from_pretrained("/mnt/d/All_Documents/documents/ViT/dataset/mantis")
    network.train()

    # 3. 收集所有唯一标签以构建编码器（只需一次）
    print("=> Building label encoder...")
    all_labels = []
    adapter_for_labels = TimeMatchToUSCropsAdapter(device='cpu')
    for batch_dict in tqdm(source_loader, desc="Scanning labels"):
        _, y_flat = adapter_for_labels(batch_dict)
        valid_y = y_flat[y_flat != -100].cpu().numpy()
        all_labels.append(valid_y)
    all_labels = np.concatenate(all_labels)
    le = LabelEncoder()
    le.fit(all_labels)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # 4. 微调 Mantis（直接在 source_loader 上训练，不创建新 DataLoader）
    print("=> Starting fine-tuning of Mantis...")
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 创建适配器（用于训练循环）
    train_adapter = TimeMatchToUSCropsAdapter(device=args.device)
    test_adapter = TimeMatchToUSCropsAdapter(device=args.device)

    for epoch in range(10):
        total_loss = 0.0
        num_batches = 0
        for batch_dict in tqdm(source_loader, desc=f"Epoch {epoch + 1}/10"):
            # 转换 batch
            X_tuple, y_flat = train_adapter(batch_dict)
            data_flat = X_tuple[0]  # [S, T, C]

            # 转换为 Mantis 输入格式 (S, C, 512)
            S, T, C = data_flat.shape
            x_mantis = data_flat.permute(0, 2, 1)  # [S, C, T]
            if T != 512:
                x_mantis = F.interpolate(x_mantis, size=512, mode='linear', align_corners=False)

            # 过滤无效样本
            valid_mask = (y_flat != -100)
            if not valid_mask.any():
                continue  # 跳过全无效 batch

            x_mantis = x_mantis[valid_mask]
            y_valid = y_flat[valid_mask]

            # 标签编码（必须在 GPU 上进行？不，先 CPU 编码再转 GPU）
            y_encoded = torch.from_numpy(le.transform(y_valid.cpu().numpy())).to(args.device)

            # 前向传播
            optimizer.zero_grad()
            logits = network(x_mantis)  # [S_valid, num_classes]
            loss = criterion(logits, y_encoded)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches > 0:
            print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / num_batches:.4f}")
        else:
            print("No valid samples in this epoch!")

    # 5. 保存模型
    save_path = f"./mantis_finetuned_{timestamp}_seed{args.seed}.pth"
    torch.save(network.state_dict(), save_path)
    print(f"Model saved to {save_path}")



    return





def main():
    args = parse_args()
    seeds = [111]
    print('seed in', seeds)
    for seed in seeds:
        # args.seed = seed
        print(f'Seed = {args.seed} --------------- ')

        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

        logdir = train(args)




if __name__ == '__main__':
    main()
