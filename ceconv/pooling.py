import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.pooling import _MaxPoolNd


class GroupMaxPool2d(_MaxPoolNd):
    r"""Applies a 2D max pooling over an input signal containing a separate group dimension and
    composed of several input planes.

    Uses the implementation of the defualt MaxPool2d layer in PyTorch.
    See:
    - https://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html#MaxPool2d
    - https://pytorch.org/docs/stable/nn.html#maxpool2d

    """

    def forward(self, x) -> torch.Tensor:
        xs = x.size()
        x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
        x = F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
        return x


class GroupCosetMaxPool(nn.Module):
    r"""Applies max pooling over the group dimension in a 5D tensor (b,c,g,w,h)."""

    def forward(self, x) -> torch.Tensor:
        return torch.max(x, dim=2)[0]


class GroupCosetAvgPool(nn.Module):
    r"""Applies max pooling over the group dimension in a 5D tensor (b,c,g,w,h)."""

    def forward(self, x) -> torch.Tensor:
        return torch.mean(x, dim=2)
