"""Unit tests for the Color Equivariant ResNet."""

import unittest

import numpy as np
import torch
from torchinfo import summary

from ceconv.ceconv2d import _get_hue_rotation_matrix
from models.resnet import ResNet18, ResNet44, ResNet50
from experiments.classification.datasets import normalize

_BATCH_SIZE = 8
_create_dummy_input = lambda: torch.rand(_BATCH_SIZE, 3, 28, 28)
_create_dummy_input_large = lambda: torch.rand(8, 3, 224, 224)


def _hue_shift_input(input: torch.Tensor, rotations: int) -> torch.Tensor:
    """Rotate input tensor in hue space."""

    hue_shift = _get_hue_rotation_matrix(rotations)
    return torch.einsum("ij,biwh->bjwh", hue_shift, input)


param_list = [
    (ResNet18, _create_dummy_input_large),
    (ResNet44, _create_dummy_input),
    (ResNet50, _create_dummy_input_large),
]


class TestResNet(unittest.TestCase):
    """Unit tests for the Color Equivariant ResNet."""

    def test_resnet(self) -> None:
        """Test the forward pass of a ResNet with different configurations."""
        for rotations in range(1, 5):
            for architecture, input_fn in param_list:
                with self.subTest(
                    architecture=architecture,
                    rotations=rotations,
                    input_fn=input_fn,
                    do_norm=False,
                    separable=False,
                ):
                    self._test_resnet(architecture, rotations, input_fn, False, False)

                    if rotations > 1:
                        self._test_resnet_cosetmax(
                            architecture, rotations, input_fn, False, False
                        )

        for rotations in range(1, 5):
            for normalize in [True, False]:
                for separable in [True, False]:
                    with self.subTest(
                        architecture=ResNet18,
                        rotations=rotations,
                        input_fn=_create_dummy_input_large,
                        do_norm=normalize,
                        separable=separable,
                    ):
                        self._test_resnet(
                            ResNet18,
                            rotations,
                            _create_dummy_input_large,
                            normalize,
                            separable,
                        )

                        if rotations > 1:
                            self._test_resnet_cosetmax(
                                ResNet18,
                                rotations,
                                _create_dummy_input_large,
                                normalize,
                                separable,
                            )

    def _test_resnet(
        self, architecture, rotations, input_fn, do_norm, separable
    ) -> None:
        """Test the forward pass of a ResNet."""

        input = input_fn()
        input_shifted = _hue_shift_input(input, 3)

        if do_norm:
            input = normalize(input, False)
            input_shifted = normalize(input_shifted, False)

        model = architecture(rotations=rotations, separable=separable, num_classes=10)
        with torch.no_grad():
            y = model(input)
            y_shifted = model(input_shifted)

        # Check output shapes.
        self.assertEqual(y.size(), (_BATCH_SIZE, 10))
        self.assertEqual(y_shifted.size(), (_BATCH_SIZE, 10))
        # Check that output is not invariant to hue shifts.
        self.assertFalse(np.allclose(y.numpy(), y_shifted.numpy(), atol=1e-5))

    def _test_resnet_cosetmax(
        self, architecture, rotations, input_fn, do_norm, separable
    ) -> None:
        """Test the forward pass of a ResNet with groupcosetmaxpool.

        The output SHOULD be invariant to hue shifts after coset pooling.
        The output SHOULD NOT be invariant to hue shifts with other rotations."""

        input = input_fn()
        input_shifted = _hue_shift_input(input, rotations)
        input_shifted_noneq = _hue_shift_input(input, rotations + 1)

        if do_norm:
            input = normalize(input, True)
            input_shifted = normalize(input_shifted, True)
            input_shifted_noneq = normalize(input_shifted_noneq, True)

        model = architecture(
            rotations=rotations,
            groupcosetmaxpool=True,
            separable=separable,
            num_classes=10,
        )
        with torch.no_grad():
            y = model(input)
            y_shifted = model(input_shifted)
            y_shifted_noneq = model(input_shifted_noneq)

        # Check output shapes.
        self.assertEqual(y.size(), (_BATCH_SIZE, 10))
        self.assertEqual(y_shifted.size(), (_BATCH_SIZE, 10))
        self.assertEqual(y_shifted_noneq.size(), (_BATCH_SIZE, 10))
        # Check output invariance.
        self.assertTrue(np.allclose(y.numpy(), y_shifted.numpy(), atol=1e-4))
        self.assertFalse(np.allclose(y.numpy(), y_shifted_noneq.numpy(), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
