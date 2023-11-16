"""Unit tests for the model module."""

import unittest
import torch

from models.cnn import CNN
from models.cnn import CECNN

_create_dummy_input = lambda: torch.rand(8, 3, 28, 28)


class TestCNN(unittest.TestCase):
    """Unit tests for the CNN class."""

    def test_forward(self) -> None:
        """Test the forward pass."""

        model = CNN(planes=32)
        y = model(_create_dummy_input())

        self.assertEqual(y.size(), (8, 10))


class TestCECNN(unittest.TestCase):
    """Unit tests for the CECNN class."""

    def test_forward(self) -> None:
        """Test the forward pass of the CECNN."""
        model = CECNN(planes=32, rotations=4)
        y = model(_create_dummy_input())
        self.assertEqual(y.shape, (8, 10))

    def test_forward_hybrid(self) -> None:
        """Test the forward pass of the CECNN."""
        model = CECNN(planes=32, rotations=4, ce_layers=4)
        y = model(_create_dummy_input())
        self.assertEqual(y.shape, (8, 10))


if __name__ == "__main__":
    unittest.main()
