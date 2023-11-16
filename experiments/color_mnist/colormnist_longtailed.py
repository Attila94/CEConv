"""Generate ColorMNIST dataset.

Generate ColorMNIST dataset with different standard deviations of the color.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST


def generate_set(dataset, samples_per_class, train) -> TensorDataset:
    """Generate 30-class color mnist dataset with long-tailed distribution."""

    imgs, targets = dataset.data.numpy(), dataset.targets.numpy()

    if train:
        # Create power law distribution for 30 classes.
        samples = np.random.power(0.3, size=imgs.shape[0]) * samples_per_class
        samples = np.ceil(samples).astype(int)
    else:
        # Create uniform distribution for 30 classes of 250 samples each.
        samples_per_class = 250
        samples = (np.ones(imgs.shape[0]) * samples_per_class).astype(int)

    # Convert to 30 classes with 3 colors per digit.
    imgs_rgb = []
    targets_rgb = []
    for i in range(10):
        samples_added = 0
        for j in range(3):
            class_idx = i * 3 + j

            # Get data.
            data_tmp = imgs[targets == i][
                samples_added : samples_added + samples[class_idx]
            ]
            # Create 3 channels and add data to j-th channel.
            data = np.zeros(data_tmp.shape + (3,))
            data[:, :, :, j] = data_tmp

            # Add data to list.
            imgs_rgb.append(data)
            targets_rgb.extend(list(np.ones(data.shape[0]) * class_idx))
            samples_added += samples[i]

    # Concatenate samples and targets.
    imgs_rgb = np.concatenate(imgs_rgb) / 255
    targets_rgb = np.asarray(targets_rgb)

    # Generate noisy background.
    ims = imgs_rgb.shape[:3]
    weight = np.max(imgs_rgb, axis=3)
    noisy_background = args.bg_intensity + np.random.randn(*ims) * args.bg_noise_std
    noisy_background = np.clip(noisy_background, 0, 1)
    # Add background to images.
    imgs_rgb = (
        weight[..., None] * imgs_rgb
        + (1 - weight[..., None]) * noisy_background[..., None]
    )

    # Convert to tensor.
    imgs_rgb = torch.from_numpy(imgs_rgb).permute(0, 3, 1, 2).float()
    targets = torch.from_numpy(targets_rgb).long()

    return TensorDataset(imgs_rgb, targets)


def generate_colormnist_longtailed(samples_per_class):
    # Create out directory.
    os.makedirs(os.environ["DATA_DIR"] + "/colormnist_longtailed", exist_ok=True)

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

    trainset = generate_set(trainset, samples_per_class, True)
    trainset_gray = TensorDataset(
        trainset.tensors[0].mean(1, keepdim=True), trainset.tensors[1]
    )
    testset = generate_set(testset, samples_per_class, False)

    # Save imagegrid.
    grid_img = torchvision.utils.make_grid(trainset.tensors[0], nrow=32)
    plt.rcParams.update({"font.size": 7})
    plt.figure(figsize=(8, 16))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    _ = plt.xticks([]), plt.yticks([])
    plt.savefig(
        os.environ["DATA_DIR"] + "/colormnist_longtailed/colormnist_longtailed.png",
        bbox_inches="tight",
    )
    plt.clf()

    grid_img = torchvision.utils.make_grid(trainset_gray.tensors[0], nrow=32)
    plt.rcParams.update({"font.size": 7})
    plt.figure(figsize=(8, 16))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    _ = plt.xticks([]), plt.yticks([])
    plt.savefig(
        os.environ["DATA_DIR"]
        + "/colormnist_longtailed/colormnist_longtailed_gray.png",
        bbox_inches="tight",
    )
    plt.clf()

    # Save datasets.
    torch.save(
        trainset,
        os.environ["DATA_DIR"] + "/colormnist_longtailed/train.pt",
    )
    torch.save(
        testset,
        os.environ["DATA_DIR"] + "/colormnist_longtailed/test.pt",
    )

    print(
        "Generated ColorMNIST - longtailed dataset at {}".format(
            os.environ["DATA_DIR"] + "/colormnist_longtailed"
        )
    )

    # Plot and save histogram of samples per class.
    samples_per_class = torch.unique(trainset.tensors[1], return_counts=True)
    sort_idx = torch.argsort(samples_per_class[1], descending=True)
    samples_per_class = (samples_per_class[0][sort_idx], samples_per_class[1][sort_idx])

    labels = [j + str(i) for i in range(10) for j in ["R", "G", "B"]]
    labels = [labels[i] for i in sort_idx.numpy()]

    plt.figure(figsize=(6, 3))
    plt.bar(labels, samples_per_class[1].numpy())
    plt.savefig(
        os.environ["DATA_DIR"] + "/colormnist_longtailed/histogram.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    """Generate dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples_per_class", type=int, default=150, help="Samples per class."
    )
    parser.add_argument(
        "--bg_noise_std", type=float, default=0.1, help="std of background noise"
    )
    parser.add_argument(
        "--bg_intensity", type=float, default=0.33, help="intensity of background"
    )
    args = parser.parse_args()

    generate_colormnist_longtailed(args.samples_per_class)
