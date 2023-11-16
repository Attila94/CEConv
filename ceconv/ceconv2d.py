"""Color Equivariant Convolutional Layer."""

import math
import typing
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def _get_hue_rotation_matrix(rotations: int) -> torch.Tensor:
    """Returns a 3x3 hue rotation matrix.

    Rotates a 3D point by 360/rotations degrees along the diagonal.

    Args:
      rotations: int, number of rotations
    """

    assert rotations > 0, "Number of rotations must be positive."

    # Constants in rotation matrix
    cos = math.cos(2 * math.pi / rotations)
    sin = math.sin(2 * math.pi / rotations)
    const_a = 1 / 3 * (1.0 - cos)
    const_b = math.sqrt(1 / 3) * sin

    # Rotation matrix
    return torch.tensor(
        [
            [cos + const_a, const_a - const_b, const_a + const_b],
            [const_a + const_b, cos + const_a, const_a - const_b],
            [const_a - const_b, const_a + const_b, cos + const_a],
        ],
        dtype=torch.float32,
    )


def _trans_input_filter(weights, rotations, rotation_matrix) -> torch.Tensor:
    """Apply linear transformation to filter.

    Args:
      weights: float32, input filter of size [c_out, 3 (c_in), 1, k, k]
      rotations: int, number of rotations applied to filter
      rotation_matrix: float32, rotation matrix of size [3, 3]
    """

    # Flatten weights tensor.
    weights_flat = weights.permute(2, 1, 0, 3, 4)  # [1, 3, c_out, k, k]
    weights_shape = weights_flat.shape
    weights_flat = weights_flat.reshape((1, 3, -1))  # [1, 3, c_out*k*k]

    # Construct full transformation matrix.
    rotation_matrix = torch.stack(
        [torch.matrix_power(rotation_matrix, i) for i in range(rotations)], dim=0
    )

    # Apply transformation to weights.
    # [rotations, 3, 3] * [1, 3, c_out*k*k] --> [rotations, 3, c_out*k*k]
    transformed_weights = torch.matmul(rotation_matrix, weights_flat)
    # [rotations, 1, c_in (3), c_out, k, k]
    transformed_weights = transformed_weights.view((rotations,) + weights_shape)
    # [c_out, rotations, c_in (3), 1, k, k]
    tw = transformed_weights.permute(3, 0, 2, 1, 4, 5)

    return tw.contiguous()


def _trans_hidden_filter(weights: torch.Tensor, rotations: int) -> torch.Tensor:
    """Perform cyclic permutation on hidden layer filter parameters."""

    # Create placeholder for output tensor
    w_shape = weights.shape
    transformed_weights = torch.zeros(
        ((w_shape[0],) + (rotations,) + w_shape[1:]), device=weights.device
    )

    # Apply cyclic permutation on output tensor
    for i in range(rotations):
        transformed_weights[:, i, :, :, :, :] = torch.roll(weights, shifts=i, dims=2)

    return transformed_weights


class CEConv2d(nn.Conv2d):
    """
    Applies a Color Equivariant convolution over an input signal composed of several
    input planes.


    Args:
        in_rotations (int): Number of input rotations: 1 for input layer, >1 for
            hidden layers.
        out_rotations (int): Number of output rotations.
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        learnable (bool): If True, the transformation matrix is learnable.
        separable (bool): If True, the convolution is separable.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_rotations: int,
        out_rotations: int,
        in_channels: int,
        out_channels: int,
        kernel_size: typing.Union[int, typing.Tuple[int, int]],
        learnable: bool = False,
        separable: bool = True,
        **kwargs
    ) -> None:
        self.in_rotations = in_rotations
        self.out_rotations = out_rotations
        self.separable = separable

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        # Initialize transformation matrix and weights.
        if in_rotations == 1:
            init = (
                torch.rand((3, 3)) * 2.0 / 3 - (1.0 / 3)
                if learnable
                else _get_hue_rotation_matrix(out_rotations)
            )
            self.transformation_matrix = Parameter(init, requires_grad=learnable)
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
            )
        else:
            if separable:
                if in_rotations > 1:
                    self.weight = Parameter(
                        # torch.Tensor(out_channels, 1, 1, *self.kernel_size)
                        torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
                    )
                    self.pointwise_weight = Parameter(
                        torch.Tensor(out_channels, in_channels, self.in_rotations, 1, 1)
                    )
            else:
                self.weight = Parameter(
                    torch.Tensor(
                        out_channels, in_channels, self.in_rotations, *self.kernel_size
                    )
                )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""

        # Compute standard deviation for weight initialization.
        n = self.in_channels * self.in_rotations * np.prod(self.kernel_size)
        stdv = 1.0 / math.sqrt(n)

        # Initialize weights.
        self.weight.data.uniform_(-stdv, stdv)
        if hasattr(self, "pointwise_weight"):
            self.pointwise_weight.data.uniform_(-stdv, stdv)

        # Initialize bias.
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        # Compute full filter weights.
        if self.in_rotations == 1:
            # Apply rotation to input layer filter.
            tw = _trans_input_filter(
                self.weight, self.out_rotations, self.transformation_matrix
            )
        else:
            # Apply cyclic permutation to hidden layer filter.
            if self.separable:
                weight = torch.mul(self.pointwise_weight, self.weight)
            else:
                weight = self.weight
            tw = _trans_hidden_filter(weight, self.out_rotations)

        tw_shape = (
            self.out_channels * self.out_rotations,
            self.in_channels * self.in_rotations,
            *self.kernel_size,
        )
        tw = tw.view(tw_shape)

        # Apply convolution.
        input_shape = input.size()
        input = input.view(
            input_shape[0],
            self.in_channels * self.in_rotations,
            input_shape[-2],
            input_shape[-1],
        )

        y = F.conv2d(
            input, weight=tw, bias=None, stride=self.stride, padding=self.padding
        )

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.out_rotations, ny_out, nx_out)

        # Apply bias.
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y
