"""Generate ColorMNIST dataset.

Generate ColorMNIST dataset with different standard deviations of the color.
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

np.random.seed(0)
torch.manual_seed(0)


def generate_set(dataset, std) -> TensorDataset:
    """Generate color mnist dataset."""

    imgs, targets = dataset.data.numpy(), dataset.targets.numpy()
    weight = imgs.copy() / 255

    # Generate random hue within range [0, 180] centered around target value.
    hue = (np.random.randn(imgs.shape[0]) * std + targets * 18) % 180

    # Add extra sv channels.
    imgs = np.stack(
        (
            hue[:, None, None] * np.ones_like(imgs),
            255 * np.ones_like(imgs),
            imgs,
        ),
        axis=3,
    ).astype("uint8")

    # Convert hsv to rgb.
    imgs_rgb = []
    for img in imgs:
        imgs_rgb.append(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
    imgs_rgb = np.stack(imgs_rgb, axis=0) / 255

    # Generate noisy background.
    ims = imgs.shape[:3]
    noisy_background = args.bg_intensity + np.random.randn(*ims) * args.bg_noise_std
    noisy_background = np.clip(noisy_background, 0, 1)
    # Add background to images.
    imgs_rgb = (
        weight[..., None] * imgs_rgb
        + (1 - weight[..., None]) * noisy_background[..., None]
    )
    imgs_rgb = np.clip(imgs_rgb, 0, 1)

    # Convert to tensor.
    imgs_rgb = torch.from_numpy(imgs_rgb).permute(0, 3, 1, 2).float()
    targets = torch.from_numpy(targets).long()

    return TensorDataset(imgs_rgb, targets)


def generate_colormnist_biased(std):
    # Create out directory.
    os.makedirs(os.environ["DATA_DIR"] + "/colormnist_biased", exist_ok=True)

    trainset = MNIST(
        root=os.environ["DATA_DIR"] + "/MNIST",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    testset = MNIST(
        root=os.environ["DATA_DIR"] + "/MNIST",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    trainset = generate_set(trainset, args.std)
    trainset_gray = TensorDataset(
        trainset.tensors[0].mean(1, keepdim=True), trainset.tensors[1]
    )
    testset = generate_set(testset, args.std)

    # Save imagegrid.
    grid_img = torchvision.utils.make_grid(trainset.tensors[0][:64, :, :, :], nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.savefig(
        os.environ["DATA_DIR"]
        + "/colormnist_biased/colormnist_biased_"
        + str(args.std)
        + ".png",
        bbox_inches="tight",
    )

    grid_img = torchvision.utils.make_grid(
        trainset_gray.tensors[0][:64, :, :, :], nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.savefig(
        os.environ["DATA_DIR"]
        + "/colormnist_biased/colormnist_biased_"
        + str(args.std)
        + "_gray.png",
        bbox_inches="tight",
    )

    # Save datasets.
    torch.save(
        trainset,
        os.environ["DATA_DIR"] + "/colormnist_biased/train_{}.pt".format(args.std),
    )
    torch.save(
        testset,
        os.environ["DATA_DIR"] + "/colormnist_biased/test_{}.pt".format(args.std),
    )

    print(
        "Generated ColorMNIST dataset with std = {} at {}".format(
            args.std, os.environ["DATA_DIR"] + "/colormnist_biased"
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--std", type=int, default=0, help="std of colormnist colors (default): 0)"
    )
    parser.add_argument(
        "--bg_noise_std", type=float, default=0.1, help="std of background noise"
    )
    parser.add_argument(
        "--bg_intensity", type=float, default=0.5, help="intensity of background"
    )
    args = parser.parse_args()

    generate_colormnist_biased(args.std)
