from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
import optuna
from torch.autograd import Variable

from torchvision import models
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict

import os


__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def load_ckpt(cfg, model):
    ckpt = cfg.model.params.ckpt
    if os.path.exists(ckpt):
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]
        # removing module if saved in lightning mode:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'model' in k:
                name = k[6:] # remove `model.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict = True)
        print('loading checkpoints weights \n')
    return model


def resnet18(cfg, **kwargs: Any):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = models.resnet18(pretrained=False, num_classes=cfg.model.nclasses)
    model = load_ckpt(cfg, model)
    return model


def resnet34(cfg, **kwargs: Any):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = models.resnet34(pretrained=False, num_classes=cfg.model.nclasses)
    model = load_ckpt(cfg, model)
    return model


def resnet50(cfg, **kwargs: Any):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = models.resnet50(pretrained=False, num_classes=cfg.model.nclasses)
    model = load_ckpt(cfg, model)
    return model
    
# pretrained=cfg.model.params.pretrained, **kwargs)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, cfg.model.nclasses)

# https://github.com/PyTorchLightning/pytorch-lightning/issues/2798
# to load state_dict from the checkpoint


if __name__ == "__main__":
    from .utils import print_info_net

    cfg = DictConfig(
        {
            "models": {
                "nclasses": 1000,
            }
        }
    )

    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)
            print_info_net(globals()[net_name](cfg))
            print()
