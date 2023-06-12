"""
Lightning Data Module class
"""
from typing import Optional, Tuple

import logging

import pytorch_lightning as pl
import torch

import glob 
import json

from collections import Counter
from pathlib import Path

from baseline.utils import load_obj
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data.distributed import Sampler, Dataset
import torch.distributed as dist

import torchvision.transforms.functional as TF
import random

T_co = TypeVar('T_co', covariant=True)

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import math

from omegaconf import DictConfig, OmegaConf

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MEANS = {
    "ImageFolder": (0.485, 0.456, 0.406),
    "PNAS": (0.7041, 0.7041, 0.7041),
    "Leaves": (0.6697, 0.6071, 0.5872),
    "SVRT": (0.98, .98, .98),
}
STDS = {
    "ImageFolder": (0.229, 0.224, 0.225),
    "PNAS": (0.2424, 0.2424, 0.2424),
    "Leaves": (0.1912, 0.2254, 0.2178),
    "SVRT": (.08, .08, .08),
}

class dataset_barense_stimuli(Dataset):
    def __init__(self, dataset_path, key, transform):
        '''
        dataset_path = contains folder with cat_1 and cat_0
        dataset_type: train, val and test
        '''
        self.dataset_path = os.path.join('/users/mvaishn1/data/data/mvaishn1/gamr_stanford/data/barense_stimuli','answer_key.json')
        self.key = key
        with open(self.dataset_path) as json_file:
            self.train_meta = json.load(json_file)

        self.file_names = self.train_meta[self.key].keys()
        self.preprocess = transform
        self.k = list(self.train_meta[self.key])
        self.v = [i for i in self.train_meta[self.key].values()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, j):
        # read image
        data_all = []
        # import pdb; pdb.set_trace()
        for idx in range(4):
            if idx == int(self.train_meta[self.key][self.k[j]])-1:
                data_path = Path(f'/users/mvaishn1/data/data/mvaishn1/gamr_stanford/data/barense_stimuli/segmented/{self.key}/{self.key}_screen{int(self.k[j]):02d}_image{idx}_oddity.bmp')
                target = idx
            else:
                data_path = Path(f'/users/mvaishn1/data/data/mvaishn1/gamr_stanford/data/barense_stimuli/segmented/{self.key}/{self.key}_screen{int(self.k[j]):02d}_image{idx}_typical.bmp')

            data = Image.open(data_path)
            data = self.preprocess(data)
            data_all.append(data)

        target = torch.tensor(target, dtype=torch.long)

        return data_all, target

class dataset_barense_stimuli_all(Dataset):
    def __init__(self, transform):
        '''
        dataset_path = contains folder with cat_1 and cat_0
        dataset_type: train, val and test
        '''
        self.dataset_path = os.path.join('/users/mvaishn1/data/data/mvaishn1/gamr_stanford/data/barense_stimuli','answer_key.json')
        self.key = ['familiar_high', 'novel_low', 'novel_high', 'familiar_low']
        with open(self.dataset_path) as json_file:
            self.train_meta = json.load(json_file)
        self.preprocess = transform
        self.file_names = []
        self.k = []
        self.v = []
        self.meta_target = []

        for k in ['familiar_high', 'novel_low', 'novel_high', 'familiar_low']:
            self.file_names+= [k for i in self.train_meta[k].keys()]
            self.k += [i for i in self.train_meta[k].keys()]
            self.v += [i for i in self.train_meta[k].values()]
            self.meta_target += [k+(i) for i in self.train_meta[k].values()]
        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, j):
        # read image
        data_all = []
        # import pdb; pdb.set_trace()
        key = self.file_names[j]
        for idx in range(4):
            if idx == int(self.train_meta[key][self.k[j]])-1:
                data_path = Path(f'/users/mvaishn1/data/data/mvaishn1/gamr_stanford/data/barense_stimuli/segmented/{key}/{key}_screen{int(self.k[j]):02d}_image{idx}_oddity.bmp')
                target = idx
            else:
                data_path = Path(f'/users/mvaishn1/data/data/mvaishn1/gamr_stanford/data/barense_stimuli/segmented/{key}/{key}_screen{int(self.k[j]):02d}_image{idx}_typical.bmp')

            data = Image.open(data_path)
            data = self.preprocess(data)
            data_all.append(data)

        target = torch.tensor(target, dtype=torch.long)

        return data_all, target

class DataModule_barense_stimuli(pl.LightningDataModule):
    def __init__(self,
                root: str = "data/",
                batch_size: int = 400,
                num_workers: int = 2,
                data_name: str = 'ODD',
                key: str = 'familiar_high',
                test_size: float = .1,
                pin_memory: bool = True,):
        super().__init__()

        means, stds = MEANS['ImageFolder'], STDS['ImageFolder']
        logger.debug(f"hard coded means: {means}, stds: {stds}")

        cfg = OmegaConf.load(os.path.join(os.getcwd(),'config.yaml'))
        self.name = data_name
        self.path = root
        self.key = key

        logger.debug(f"Dataset path is: {self.path}")
        
        self.root_train = to_absolute_path(root)        
        self.batch_size = batch_size
        self.num_workers =  num_workers  
        self.pin_memory = pin_memory

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.RandomAutocontrast(),
                # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)
            ]
        )

        train_full_data = dataset_barense_stimuli(self.root_train, key = self.key, \
            transform=self.train_transforms)
        
        # import pdb; pdb.set_trace()
        targets = [int(i) for i in train_full_data.v]
        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), \
                                              stratify=targets, test_size=test_size, \
                                                shuffle=True)
        print(len(val_idx))
        print(len(train_idx))
        self.train_data = Subset(train_full_data, train_idx)
        self.val_data = Subset(train_full_data, val_idx)


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)

class DataModule_barense_stimuli_all(pl.LightningDataModule):
    def __init__(self,
                root: str = "data/",
                batch_size: int = 400,
                num_workers: int = 2,
                data_name: str = 'ODD',
                key: str = 'familiar_high',
                test_size: float = .1,
                pin_memory: bool = True,):
        super().__init__()

        means, stds = MEANS['ImageFolder'], STDS['ImageFolder']
        logger.debug(f"hard coded means: {means}, stds: {stds}")

        cfg = OmegaConf.load(os.path.join(os.getcwd(),'config.yaml'))
        self.name = data_name
        self.path = root
        self.key = key

        logger.debug(f"Dataset path is: {self.path}")
        
        self.root_train = to_absolute_path(root)        
        self.batch_size = batch_size
        self.num_workers =  num_workers  
        self.pin_memory = pin_memory

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.RandomAutocontrast(),
                # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)
            ]
        )

        train_full_data = dataset_barense_stimuli_all(transform=self.train_transforms)
        
        # import pdb; pdb.set_trace()
        targets = [i for i in train_full_data.meta_target]
        train_idx, val_idx = train_test_split(list(range(len(train_full_data))), \
                                              stratify=targets, test_size=test_size, \
                                                shuffle=True)
        print(len(val_idx))
        print(len(train_idx))
        self.train_data = Subset(train_full_data, train_idx)
        self.val_data = Subset(train_full_data, val_idx)


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)