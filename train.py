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
import numpy as np
from pathlib import Path
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random

from datetime import datetime
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

import torch
def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models.')
    parser.add_argument('-n', '--per', default=0.3, type=int,
                        help='percentage of labeled samples (training and validation) (default )')
    parser.add_argument('--seed', default=111, type=int,
                        help='seed')
    # 以下都是timematch
    parser.add_argument('--gpus', type=int, default=4,
                        help='Number of GPUs to use (0=CPU, 1=Single GPU, >=2=Multi-GPU DDP)')
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of workers"
    )
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--balance_source", type=bool_flag, default=True, help='class balanced batches for source')
    parser.add_argument('--num_pixels', default=50, type=int, help='Number of pixels to sample from the input sample')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of time steps to sample from the input sample')
    # 数据路径与域
    parser.add_argument('--data_root', default='/data/user/DBL/timematch_data', type=str,
                        help='Path to datasets root directory')
    # parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str,
    #                     help='Path to datasets root directory')
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
    if args.useall or args.per ==1 :
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
