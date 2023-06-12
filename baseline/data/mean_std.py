from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch

root = '/cifs/data/tserre_lrs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100/'
# /cifs/data/tserre_lrs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/512/train

train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

train_data = datasets.ImageFolder(os.path.join(root, 'train'),
            transform=train_transforms)

loader = torch.utils.data.DataLoader(train_data,
                                     batch_size = 100,
                                     num_workers = 0,
                                     shuffle = False)

image_means = []
image_std = []
for t, c in train_data:
    image_means.append(t.mean(1).mean(1))
    image_std.append(t.std(1).mean(1))

image_means = torch.stack(image_means)
print(image_means.mean(0))

image_std = torch.stack(image_std)
print(image_std.mean(0))

# mean: [0.7041, 0.7041, 0.7041]
# std: [0.2424, 0.2424, 0.2424]
'''
for 1024:
    tensor([0.7027, 0.7027, 0.7027])
    tensor([0.2347, 0.2347, 0.2347])
'''