"""Unit tests for ceconv2d.py."""

import numpy as np
import torch
import unittest

from ceconv.ceconv2d import CEConv2d
from ceconv.ceconv2d import _get_hue_rotation_matrix
from ceconv.ceconv2d import _trans_input_filter
from experiments.classification.datasets import normalize

_BATCH_SIZE = 8
_create_dummy_input = lambda: torch.rand(_BATCH_SIZE, 3, 32, 32)


def _hue_shift_input(input: torch.Tensor, rotations: int) -> torch.Tensor:
    """Rotate input tensor in hue space."""

    hue_shift = _get_hue_rotation_matrix(rotations)
    return torch.einsum("ij,biwh->bjwh", hue_shift, input)


class CEConv2DTest(unittest.TestCase):
    """Unit tests for CEConv2D and helper functions."""

    def test_transformation_matrix(self) -> None:
        """Test the transformation matrix for hue rotations."""

        for rotations in range(2, 5):
            with self.subTest(rotations=rotations):
                self._test_get_hue_rotation_matrix_identity(rotations)
                self._test_get_hue_rotation_matrix_inverse(rotations)
                self._test_trans_input_filter(rotations)

    def test_ceconv2d(self) -> None:
        """Test the forward pass of a CEConv2D layer."""

        for rotations in range(2, 5):
            for separable in [True, False]:
                with self.subTest(rotations=rotations, separable=separable):
                    self._test_output_size(rotations, separable)
                    self._test_equivariance(rotations, separable)
                    self._test_equivariance_normalized(rotations, separable)

    def _test_get_hue_rotation_matrix_identity(self, rotations) -> None:
        """Matrix for i rotations to the power of i should be the identity matrix."""

        self.assertTrue(
            np.allclose(
                np.linalg.matrix_power(_get_hue_rotation_matrix(rotations), rotations),
                np.eye(3),
                atol=1e-6,
            )
        )

    def _test_get_hue_rotation_matrix_inverse(self, rotations) -> None:
        """Transpose of matrix should be the inverse."""

        self.assertTrue(
            np.allclose(
                np.linalg.inv(_get_hue_rotation_matrix(rotations)),
                _get_hue_rotation_matrix(rotations).T,
                atol=1e-6,
            )
        )

    def _test_trans_input_filter(self, rotations) -> None:
        """Check transformed input filter."""

        kernel_size = 3
        in_channels = 3
        out_channels = 4

        # Define dummy filter weight of shape
        # [out_channels, in_channels, 1, kernel_size, kernel_size].
        filter_weight = torch.rand(4, 3, 1, 3, 3)

        # Get rotation matrix.
        rotation_matrix = _get_hue_rotation_matrix(rotations)

        # Apply filter transformation.
        transformed_filter_weight = _trans_input_filter(
            filter_weight, rotations, rotation_matrix
        )

        # Verify filter weight is of shape
        # [out_channels, out_rotations, in_channels, 1, kernel_size, kernel_size].
        self.assertEqual(
            transformed_filter_weight.size(),
            (out_channels, rotations, in_channels, 1, kernel_size, kernel_size),
            msg="Filter weight has wrong shape.",
        )

        # First rotation should be equal to original filter.
        self.assertTrue(
            np.allclose(
                transformed_filter_weight[:, 0, :, :, :, :].detach().numpy(),
                filter_weight,
            ),
            msg="First rotation should be equal to original filter.",
        )

        # For 3 rotations, the filter should contain permuted copies in the new dimension.
        if rotations == 3:
            self.assertTrue(
                np.allclose(
                    transformed_filter_weight[0, 0, :, 0, :, :],
                    torch.roll(
                        transformed_filter_weight[0, 1, :, 0, :, :], shifts=-1, dims=0
                    ),
                    atol=1e-6,
                )
            )

    def _test_output_size(self, rotations, separable) -> None:
        """Check output sizes for lifting and hidden layers."""

        conv1 = CEConv2d(
            1, rotations, 3, 16, 3, bias=True, padding=1, separable=separable
        )
        conv2 = CEConv2d(
            rotations, rotations, 16, 32, 3, bias=True, padding=1, separable=separable
        )

        output1 = conv1(_create_dummy_input())
        self.assertEqual(output1.size(), (_BATCH_SIZE, 16, rotations, 32, 32))

        output2 = conv2(output1)
        self.assertEqual(output2.size(), (_BATCH_SIZE, 32, rotations, 32, 32))

    def _test_equivariance(self, rotations, separable) -> None:
        """Check equivariance property of CEConv."""

        # Define color equivariant network.
        net = torch.nn.Sequential(
            CEConv2d(1, rotations, 3, 16, 3, bias=True, separable=separable),
            CEConv2d(rotations, rotations, 16, 32, 3, bias=True, separable=separable),
        )

        # Generate random input and hue-rotate it.
        input = _create_dummy_input()
        input_shifted = _hue_shift_input(input, rotations)

        # Forward pass through network.
        with torch.no_grad():
            output = net(input)
            output_shifted = net(input_shifted)

        self.assertTrue(
            np.allclose(
                output.numpy(),
                torch.roll(output_shifted, shifts=1, dims=2).numpy(),
                atol=1e-6,
            )
        )

    def _test_equivariance_normalized(self, rotations, separable) -> None:
        """Check equivariance property of CEConv."""

        # Define color equivariant network.
        net = torch.nn.Sequential(
            CEConv2d(1, rotations, 3, 16, 3, bias=True, separable=separable),
            CEConv2d(rotations, rotations, 16, 32, 3, bias=True, separable=separable),
        )

        # Generate random input and hue-rotate it.
        input = _create_dummy_input()
        input_shifted = _hue_shift_input(input, rotations)

        # Normalize inputs.
        input = normalize(input, True)
        input_shifted = normalize(input_shifted, True)

        # Forward pass through network.
        with torch.no_grad():
            output = net(input)
            output_shifted = net(input_shifted)

        self.assertTrue(
            np.allclose(
                output.numpy(),
                torch.roll(output_shifted, shifts=1, dims=2).numpy(),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
