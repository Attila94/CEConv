"""Hybrid Color Equivariant ResNet in PyTorch with mixed CEConv2D and nn.Conv2d layers.

`ce_stages` parameters controls the number of stages where CEConv2D is used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from ceconv.ceconv2d import CEConv2d
from ceconv.pooling import GroupCosetMaxPool, GroupMaxPool2d

tv_pth = "https://download.pytorch.org/models/"
hf_pth = "https://huggingface.co/attilalengyel/ceresnet/resolve/main/"

model_urls = {
    # (layers, rotations, ce_stages, groupcosetmaxpool, separable)
    # torchhub
    ("resnet18", 1, 0, False, False): tv_pth + "resnet18-5c106cde.pth",
    ("resnet34", 1, 0, False, False): tv_pth + "resnet34-333f7ec4.pth",
    ("resnet50", 1, 0, False, False): tv_pth + "resnet50-19c8e357.pth",
    ("resnet101", 1, 0, False, False): tv_pth + "resnet101-5d3b4d8f.pth",
    ("resnet152", 1, 0, False, False): tv_pth + "resnet152-b121ed2d.pth",
    # huggingface
    ("resnet18_jitter", 1, 0, False, False): hf_pth + "resnet18-1-jitter.pth",
    ("resnet18", 3, 4, False, True): hf_pth + "resnet18-3-false.pth",
    ("resnet18", 3, 4, True, False): hf_pth + "resnet18-3-true-nonsep.pth",
    ("resnet18", 3, 4, True, True): hf_pth + "resnet18-3-true.pth",
    ("resnet18", 3, 3, True, True): hf_pth + "resnet18-3-true-h3.pth",
    ("resnet18", 3, 2, True, True): hf_pth + "resnet18-3-true-h2.pth",
    ("resnet18", 3, 1, True, True): hf_pth + "resnet18-3-true-h1.pth",
    ("resnet18_jitter", 3, 4, True, True): hf_pth + "resnet18-3-true-jitter.pth",
    ("resnet18_jitter", 3, 3, True, True): hf_pth + "resnet18-3-true-jitter-h3.pth",
    ("resnet18_jitter", 3, 2, True, True): hf_pth + "resnet18-3-true-jitter-h2.pth",
    ("resnet18_jitter", 3, 1, True, True): hf_pth + "resnet18-3-true-jitter-h1.pth",
    ("resnet50", 3, 4, True, True): hf_pth + "resnet50-3-true.pth",
}


def convert_keys(model_dict: dict):
    """Convert Torchvision pretrained ResNet keys."""
    model_dict_new = {}
    for k, v in model_dict.items():
        k = k.replace("layer1", "layers.0")
        k = k.replace("layer2", "layers.1")
        k = k.replace("layer3", "layers.2")
        k = k.replace("layer4", "layers.3")
        k = k.replace("downsample", "shortcut")
        k = k.replace("fc", "linear")
        model_dict_new[k] = v
    return model_dict_new


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, rotations=1, separable=False
    ) -> None:
        super(BasicBlock, self).__init__()

        bnlayer = nn.BatchNorm2d if rotations == 1 else nn.BatchNorm3d
        self.bn1 = bnlayer(planes)
        self.bn2 = bnlayer(planes)

        self.shortcut = nn.Sequential()

        self.stride = stride
        self.kernel_size = 3
        self.padding = 2

        if rotations == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    bnlayer(self.expansion * planes),
                )
        else:
            self.conv1 = CEConv2d(
                rotations,
                rotations,
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                separable=separable,
            )
            self.conv2 = CEConv2d(
                rotations,
                rotations,
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                separable=separable,
            )
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    CEConv2d(
                        rotations,
                        rotations,
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        separable=False,
                    ),
                    bnlayer(self.expansion * planes),
                )

    def forward(self, x) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, rotations=1, separable=False):
        super(Bottleneck, self).__init__()
        bnlayer = nn.BatchNorm2d if rotations == 1 else nn.BatchNorm3d
        self.bn1 = bnlayer(planes)
        self.bn2 = bnlayer(planes)
        self.bn3 = bnlayer(self.expansion * planes)

        self.shortcut = nn.Sequential()

        if rotations == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.conv3 = nn.Conv2d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )

            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    bnlayer(self.expansion * planes),
                )
        else:
            self.conv1 = CEConv2d(
                rotations,
                rotations,
                in_planes,
                planes,
                kernel_size=1,
                bias=False,
                separable=separable,
            )
            self.conv2 = CEConv2d(
                rotations,
                rotations,
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                separable=separable,
            )
            self.conv3 = CEConv2d(
                rotations,
                rotations,
                planes,
                self.expansion * planes,
                kernel_size=1,
                bias=False,
                separable=separable,
            )

            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    CEConv2d(
                        rotations,
                        rotations,
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        separable=False,
                    ),
                    bnlayer(self.expansion * planes),
                )

    def forward(self, x) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HybridResNet(nn.Module):
    """ResNet with CEConv2D and Conv2d layers.

    Args:
        block (nn.Module): BasicBlock or Bottleneck.
        num_blocks (list): Number of blocks per stage.
        ce_stages (int): Number of stages to use CEConv2D. If 0, use only Conv2d.
            If empty or greater than len(num_blocks), use CEConv2D in all stages.
        num_classes (int): Number of classes for final FC layer.
        rotations (int): Number of rotations to use for CEConv2D.
        groupcosetmaxpool (bool): Use coset max pooling. Must be true for hybrid
            models with both CEConv2D and Conv2d layers.
        learnable (bool): Use learnable rotations.
        width (int): Width of first stage.
        separable (bool): Use separable convolutions.
    """

    def __init__(
        self,
        block,
        num_blocks,
        ce_stages=99,
        num_classes=1000,
        rotations=1,
        groupcosetmaxpool=False,
        learnable=False,
        width=64,
        separable=False,
    ) -> None:
        super(HybridResNet, self).__init__()

        assert rotations > 0, "rotations must be greater than 0"
        if rotations > 1:
            assert ce_stages > 0, "ce_stages must be greater than 0"

        if groupcosetmaxpool == False and ce_stages < len(num_blocks) and ce_stages > 0:
            raise NotImplementedError(
                "Intermediate flattening not implemented, use GroupCosetMaxPool"
            )

        # Compute channels per stage.
        # If hybrid model, assume channel scaling is done in width argument.
        if ce_stages > 0 and ce_stages < len(num_blocks):
            channels = [width * 2**i for i in range(len(num_blocks))]
        # Else, set width to default and compute scaling based on rotations.
        else:
            width = 64 if len(num_blocks) == 4 else 32
            if separable and rotations > 1:
                channels = [
                    math.floor(math.sqrt(9 * width**2 / (9 + rotations)) * 2**i)
                    for i in range(len(num_blocks))
                ]
            else:
                channels = [
                    int(width / math.sqrt(rotations) * 2**i)
                    for i in range(len(num_blocks))
                ]

        self.ce_stages = [i < ce_stages for i in range(len(num_blocks))] + [False]

        self.in_planes = channels[0]
        strides = [1, 2, 2, 2]

        # Adjust 3-stage architectures for low-res input, e.g. cifar.
        low_resolution = True if len(num_blocks) == 3 else False
        conv1_kernelsize = 3 if low_resolution else 7
        conv1_stride = 1 if low_resolution else 2
        self.maxpool = nn.Identity()

        # Use CEConv2D for rotations > 1.
        if rotations > 1:
            self.conv1 = CEConv2d(
                1,  # in_rotations
                rotations,
                3,  # in_channels
                channels[0],
                kernel_size=conv1_kernelsize,
                stride=conv1_stride,
                padding=1,
                bias=False,
                learnable=learnable,
                separable=separable,
            )
            self.bn1 = nn.BatchNorm3d(channels[0])
            if not low_resolution:
                self.maxpool = GroupMaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(
                3,
                channels[0],
                kernel_size=conv1_kernelsize,
                stride=conv1_stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(channels[0])
            if not low_resolution:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Build resblocks
        self.layers = nn.ModuleList([])
        for i in range(len(num_blocks)):
            block_rotations = rotations if self.ce_stages[i] else 1
            self.layers.append(
                self._make_layer(
                    block,
                    channels[i],
                    num_blocks[i],
                    stride=strides[i],
                    rotations=block_rotations,
                    separable=separable,
                )
            )
        # Pooling layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cosetpoollayer = None
        if groupcosetmaxpool:
            self.linear = nn.Linear(channels[-1] * block.expansion, num_classes)
            self.cosetpoollayer = GroupCosetMaxPool()
        else:
            self.linear = nn.Linear(
                channels[-1] * rotations * block.expansion, num_classes
            )

    def _make_layer(self, block, planes, num_blocks, stride, rotations, separable):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rotations, separable))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        for i, layer in enumerate(self.layers):
            out = layer(out)

            # Pool or flatten between CE and non-CE stages.
            outs = out.shape
            if self.ce_stages[i + 1] is False and len(out.shape) == 5:
                if self.cosetpoollayer is not None:
                    out = self.cosetpoollayer(out)
                else:
                    out = out.view(outs[0], -1, outs[-2], outs[-1])

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _HybridResNet(arch, block, layers, pretrained, progress, jitter=False, **kwargs):
    model = HybridResNet(block, layers, **kwargs)
    if pretrained:
        # Key is tuple of (layers, rotations, groupcosetmaxpool, separable)
        if jitter:
            arch += "_jitter"
        key = (
            arch,
            kwargs["rotations"],
            kwargs.get("ce_stages", 4),
            kwargs.get("groupcosetmaxpool", False),
            kwargs.get("separable", False),
        )
        if key not in model_urls:
            raise ValueError(
                "No checkpoint is available for this model with the given parameters."
            )
        state_dict = load_state_dict_from_url(model_urls[key], progress=progress)
        r = model.load_state_dict(convert_keys(state_dict))
        print(r)
    return model


def HybridResNet18(pretrained=False, progress=True, **kwargs):
    """ResNet18 baseline (width=64, num_classes=1000) = 11,689,512
    Separable TRUE:
        - ce_stages=1 --> width=63
        - ce_stages=2 --> width=63
        - ce_stages=3 --> width=61
        - ce_stages=4 --> width=55
    Separable FALSE:
        - ce_stages=1 --> width=63
        - ce_stages=2 --> width=60
        - ce_stages=3 --> width=52
        - ce_stages=4 --> width=37
    """
    width_dict = {
        (True, 0): 64,
        (True, 1): 63,
        (True, 2): 63,
        (True, 3): 61,
        (True, 4): 55,
        (False, 0): 64,
        (False, 1): 63,
        (False, 2): 60,
        (False, 3): 52,
        (False, 4): 37,
    }
    # If width is not specified, use the width from the above dictionary.
    w = width_dict[(kwargs.get("separable", True), kwargs.get("ce_stages", 4))]  # type: ignore
    kwargs["width"] = kwargs.get("width", w)
    return _HybridResNet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def HybridResNet34(pretrained=False, progress=True, **kwargs):
    return _HybridResNet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def HybridResNet44(pretrained=False, progress=True, **kwargs):
    """ResNet44 baseline (width=32, num_classes=10) = 2,636,458
    Separable TRUE:
        - ce_stages=1 --> width=31
        - ce_stages=2 --> width=30
        - ce_stages=3 --> width=27
    Separable FALSE:
        - ce_stages=1 --> width=30
        - ce_stages=2 --> width=26
        - ce_stages=3 --> width=18
    """
    width_dict = {
        (True, 0): 32,
        (True, 1): 31,
        (True, 2): 30,
        (True, 3): 27,
        (False, 0): 32,
        (False, 1): 30,
        (False, 2): 26,
        (False, 3): 18,
    }
    # If width is not specified, use the width from the above dictionary.
    w = width_dict[(kwargs.get("separable", True), kwargs.get("ce_stages", 3))]  # type: ignore
    kwargs["width"] = kwargs.get("width", w)
    return _HybridResNet(
        "resnet44", BasicBlock, [7, 7, 7], pretrained, progress, **kwargs
    )


def HybridResNet50(pretrained=False, progress=True, **kwargs):
    # TODO: Fix auto-scaling of width for pretrained networks.
    if pretrained:
        kwargs["width"] = 47
        kwargs["separable"] = True
        kwargs["groupcosetmaxpool"] = True
        print(
            "WARNING - following parameters have been set automatically for pretrained "
            "network: \n"
            "\t width=47, separable=True, groupcosetmaxpool=True"
        )
    return _HybridResNet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def HybridResNet101(pretrained=False, progress=True, **kwargs):
    return _HybridResNet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def HybridResNet152(pretrained=False, progress=True, **kwargs):
    return _HybridResNet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


if __name__ == "__main__":
    from torchinfo import summary

    summary(
        HybridResNet18(
            rotations=3,
            ce_stages=2,
            separable=True,
            groupcosetmaxpool=True,
            pretrained=True,
            jitter=False,
        ),
        (2, 3, 224, 224),
        device="cpu",
    )
