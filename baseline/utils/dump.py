from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
import optuna
from torch.autograd import Variable

from torchvision.models.utils import load_state_dict_from_url


from .base import LitClassifier, LitClassifier_optuna, LitClassifier_op

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet50v1", 
    "resnet50v2", 
    "resnet50v3", 
    "resnet50v4", 
    "resnet50v5", 
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
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

def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(LitClassifier_op):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        cfg: DictConfig,
        zero_init_residual: bool = False,
        groups: int = 1,
        num_classes = 1000,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(cfg)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes #cfg.model.nclasses

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetv1(ResNet):
    def __init__(self,block,layers,cfg):
        super().__init__(block,layers,cfg)
        # v1 version without alpha
        self.a1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a4 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.fc = nn.Linear((512 * block.expansion)+512+256+1024, self.num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        c4 = torch.flatten(self.avgpool(x4),1)
        c3 = torch.flatten(self.avgpool(x3),1)
        c2 = torch.flatten(self.avgpool(x2),1)
        c1 = torch.flatten(self.avgpool(x1),1)
        x = torch.cat([self.a1.type_as(x)*c4,self.a2.type_as(x)*c3,self.a3.type_as(x)*c2,self.a4.type_as(x)*c1], 1)
        # x = torch.cat([c4,c3,c2,c1], 1)
        # x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetv2(ResNet):
    def __init__(self,block,layers,cfg):
        super().__init__(block,layers,cfg)

        # v1 version without alpha
        self.a1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a4 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.fc1 = nn.Linear((512 * block.expansion)+512+256+1024, 2048)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2048, self.num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        c4 = torch.flatten(self.avgpool(x4),1)
        c3 = torch.flatten(self.avgpool(x3),1)
        c2 = torch.flatten(self.avgpool(x2),1)
        c1 = torch.flatten(self.avgpool(x1),1)
        x = torch.cat([self.a1.type_as(x)*c4,self.a2.type_as(x)*c3,self.a3.type_as(x)*c2,self.a4.type_as(x)*c1], 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class ResNetv3(ResNet):
    def __init__(self,block,layers,cfg, num_classes):
        super().__init__(block,layers,cfg, num_classes)
        
        self.a1 = nn.Parameter(torch.tensor(.33), requires_grad=True)
        self.a2 = nn.Parameter(torch.tensor(.33), requires_grad=True)
        self.a3 = nn.Parameter(torch.tensor(.33), requires_grad=True)
        self.bn_l = nn.BatchNorm1d(num_features=2048)
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        '''
        Start with lowest dimension
        '''

        # basic block
        x = self.conv1(x) # input = 1024
        x = self.bn1(x)
        x = self.relu(x)
        x_b1 = self.maxpool(x) # out = 512,
        x_b2 = self.maxpool(x_b1) # out = 256,
        x_b3 = self.maxpool(x_b2) # out = 128,

        o_b1 = F.relu(self.bn_l(self._forward_impl(x_b1)))
        o_b2 = F.relu(self.bn_l(self._forward_impl(x_b2.clone().detach())))
        o_b3 = F.relu(self.bn_l(self._forward_impl(x_b3.clone().detach())))

        x = self.a1.type_as(x)*o_b1+self.a2.type_as(x)*o_b2+self.a3.type_as(x)*o_b3

        # x = o_b1+o_b2+o_b3

        x = self.fc(x)

        return x

class ResNetv4(ResNet):
    def __init__(self,block,layers,cfg, num_classes):
        super().__init__(block,layers,cfg, num_classes)
        
        norm_layer = nn.BatchNorm2d
        self.dropout = nn.Dropout(p=0.4)
        self.dropout2d = nn.Dropout2d(p=0.2)

        # for scale I/4:
        self.conv1_s4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1_s4 = norm_layer(512)
        self.relu1_s4 = nn.ReLU(inplace=True)

        # self.conv2_s4 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn2_s4 = norm_layer(1024)
        # self.relu2_s4 = nn.ReLU(inplace=True)

        # for scale I/2:
        self.conv1_s2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1_s2 = norm_layer(512)
        self.relu1_s2 = nn.ReLU(inplace=True)

        self.conv2_s2 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2_s2 = norm_layer(1024)
        self.relu2_s2 = nn.ReLU(inplace=True)

        # self.conv3_s2 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn3_s2 = norm_layer(2048)
        # self.relu3_s2 = nn.ReLU(inplace=True)

        # for dealiasing, I/4:
        self.conv1_da_s4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_da_s4 = norm_layer(512)
        self.relu1_da_s4 = nn.ReLU(inplace=True)

        # self.conv2_da_s4 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn2_da_s4 = norm_layer(1024)
        # self.relu2_da_s4 = nn.ReLU(inplace=True)

        # for dealiasing, I/2:
        self.conv1_da_s2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_da_s2 = norm_layer(512)
        self.relu1_da_s2 = nn.ReLU(inplace=True)

        self.conv2_da_s2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_da_s2 = norm_layer(1024)
        self.relu2_da_s2 = nn.ReLU(inplace=True)

        # self.conv3_da_s2 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn3_da_s2 = norm_layer(2048)
        # self.relu3_da_s2 = nn.ReLU(inplace=True)

    def _forward_impl_s4(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        
        # x = x.clone().detach()
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
            
        return x1, x2
    
    def _forward_impl_s2(self, x: Tensor, o1, o2) -> Tensor:
        # See note [TorchScript super()]
        
        # x = x.clone().detach()
        x1 = self.layer1(x)

        x2 = self.layer2(x1) + self.relu1_s4(self.bn1_s4(self.conv1_s4(o1)))
        x2 = self.relu1_da_s4(self.bn1_da_s4(self.conv1_da_s4(x2)))
        
        # x3 = self.layer3(x2) + self.relu2_s4(self.bn2_s4(self.conv2_s4(o2)))
        # x3 = self.relu2_da_s4(self.bn2_da_s4(self.conv2_da_s4(x3)))
        
        return x1, x2 #, x3

    def _forward_impl(self, x: Tensor, o1, o2) -> Tensor:
        # See note [TorchScript super()]
        
        x = self.layer1(x)

        x = self.layer2(x) + self.dropout2d(self.relu1_s2(self.bn1_s2(self.conv1_s2(o1))))
        x = self.relu1_da_s2(self.bn1_da_s2(self.conv1_da_s2(x)))
        
        x = self.layer3(x) + self.dropout2d(self.relu2_s2(self.bn2_s2(self.conv2_s2(o2))))
        x = self.relu2_da_s2(self.bn2_da_s2(self.conv2_da_s2(x)))

        x = self.layer4(x) # + self.relu3_s2(self.bn3_s2(self.conv3_s2(o3)))
        # x = self.relu3_da_s2(self.bn3_da_s2(self.conv3_da_s2(x)))

        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        '''
        Start with lowest dimension
        '''

        # basic block
        x = self.conv1(x) # input = 1024
        x = self.bn1(x)
        x = self.relu(x)
        x_b1 = self.maxpool(x) # out = 512,
        x_b2 = self.maxpool(x_b1) # out = 256,
        x_b3 = self.maxpool(x_b2) # out = 128,

        o_b1_x1, o_b1_x2 = self._forward_impl_s4(x_b3)
        o_b2_x1, o_b2_x2 = self._forward_impl_s2(x_b2, o_b1_x1, o_b1_x2)
        x = self._forward_impl(x_b1, o_b2_x1, o_b2_x2) # , o_b2_x3)
        x = self.fc(self.dropout(x))

        return x

class ResNetv5(ResNet):
    def __init__(self,block,layers,cfg, num_classes):
        super().__init__(block,layers,cfg, num_classes)

        self.dropout = nn.Dropout(p=0.4)
        self.dropout2d = nn.Dropout2d(p=0.2)
        norm_layer = nn.BatchNorm2d

        self.conv1_s2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1_s2 = norm_layer(512)
        self.relu1_s2 = nn.ReLU(inplace=True)

        self.conv2_s2 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2_s2 = norm_layer(1024)
        self.relu2_s2 = nn.ReLU(inplace=True)

        # self.conv3_s2 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn3_s2 = norm_layer(2048)
        # self.relu3_s2 = nn.ReLU(inplace=True)

        # for dealiasing, I/2:
        self.conv1_da_s2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_da_s2 = norm_layer(512)
        self.relu1_da_s2 = nn.ReLU(inplace=True)

        self.conv2_da_s2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_da_s2 = norm_layer(1024)
        self.relu2_da_s2 = nn.ReLU(inplace=True)

        # self.conv3_da_s2 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn3_da_s2 = norm_layer(2048)
        # self.relu3_da_s2 = nn.ReLU(inplace=True)

    def _forward_impl_s2(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)         
        x3 = self.layer3(x2) 
        
        return x1, x2, x3

    def _forward_impl(self, x: Tensor, o1, o2, o3) -> Tensor:
        # See note [TorchScript super()]
        
        x = self.layer1(x)

        x = self.layer2(x) + self.dropout2d(self.relu1_s2(self.bn1_s2(self.conv1_s2(o1))))
        x = self.relu1_da_s2(self.bn1_da_s2(self.conv1_da_s2(x)))
        
        x = self.layer3(x) + self.dropout2d(self.relu2_s2(self.bn2_s2(self.conv2_s2(o2))))
        x = self.relu2_da_s2(self.bn2_da_s2(self.conv2_da_s2(x)))

        x = self.layer4(x) #+ self.relu3_s2(self.bn3_s2(self.conv3_s2(o3)))
        # x = self.relu3_da_s2(self.bn3_da_s2((x)))

        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        '''
        Start with lowest dimension
        '''

        # basic block
        x = self.conv1(x) # input = 1024
        x = self.bn1(x)
        x = self.relu(x)
        x_b1 = self.maxpool(x) # out = 512,
        x_b2 = self.maxpool(x_b1) # out = 256,

        o_b2_x1, o_b2_x2, o_b2_x3 = self._forward_impl_s2(x_b2)
        x = self._forward_impl(x_b1, o_b2_x1, o_b2_x2, o_b2_x3)
        x = self.fc(self.dropout(x))

        return x


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    cfg: DictConfig,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, cfg, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model

def _resnetv1(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    cfg: DictConfig,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> ResNetv1:
    model = ResNetv1(block, layers, cfg, **kwargs)
    
    return model

def _resnetv2(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    cfg: DictConfig,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> ResNetv2:
    model = ResNetv2(block, layers, cfg, **kwargs)
    
    return model

def _resnetv3(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    cfg: DictConfig,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> ResNetv3:
    model = ResNetv3(block, layers, cfg, num_classes, **kwargs)
    
    return model

def _resnetv4(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    cfg: DictConfig,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> ResNetv4:
    model = ResNetv4(block, layers, cfg, num_classes, **kwargs)
    
    return model

def _resnetv5(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    cfg: DictConfig,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> ResNetv4:
    model = ResNetv5(block, layers, cfg, num_classes, **kwargs)
    
    return model

def resnet18(cfg, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], cfg, **kwargs)


def resnet34(cfg, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], cfg, **kwargs)


def resnet50(cfg, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    
    if cfg.model.nclasses == 1000:
        model = _resnet("resnet50", Bottleneck, [3, 4, 6, 3], cfg, num_classes=1000, \
                         pretrained=cfg.model.params, **kwargs)
        return model

    else:
        model = _resnet("resnet50", Bottleneck, [3, 4, 6, 3], cfg, num_classes=1000, \
                        pretrained=cfg.model.params, **kwargs)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cfg.model.nclasses)
        return model

# https://github.com/PyTorchLightning/pytorch-lightning/issues/2798
# to load state_dict from the checkpoint

def resnet50v1(cfg, **kwargs: Any) -> ResNetv1:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = _resnetv1("resnet50v1", Bottleneck, [3, 4, 6, 3], cfg, num_classes=1000, \
                    pretrained=cfg.model.params, **kwargs)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.model.nclasses)
    return model

def resnet50v2(cfg, **kwargs: Any) -> ResNetv1:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = _resnetv2("resnet50v2", Bottleneck, [3, 4, 6, 3], cfg, num_classes=1000, \
                    pretrained=cfg.model.params, **kwargs)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.model.nclasses)
    return model

def resnet50v3(cfg, **kwargs: Any) -> ResNetv1:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = _resnetv3("resnet50v3", Bottleneck, [3, 4, 6, 3], cfg, num_classes=cfg.model.nclasses, \
                    pretrained=cfg.model.params, **kwargs)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.model.nclasses)
    return model

def resnet50v4(cfg, **kwargs: Any) -> ResNetv1:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = _resnetv4("resnet50v4", Bottleneck, [3, 4, 6, 3], cfg, num_classes=cfg.model.nclasses, \
                    pretrained=cfg.model.params, **kwargs)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.model.nclasses)
    return model

def resnet50v5(cfg, **kwargs: Any) -> ResNetv1:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = _resnetv5("resnet50v5", Bottleneck, [3, 4, 6, 3], cfg, num_classes=cfg.model.nclasses, \
                    pretrained=cfg.model.params, **kwargs)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.model.nclasses)
    return model

def resnet101(cfg, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], cfg, **kwargs)


def resnet152(cfg, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], cfg, **kwargs)


def resnext50_32x4d(cfg, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], cfg, **kwargs)


def resnext101_32x8d(cfg, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], cfg, **kwargs)


def wide_resnet50_2(cfg, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], cfg, **kwargs)


def wide_resnet101_2(cfg, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], cfg, **kwargs)


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
