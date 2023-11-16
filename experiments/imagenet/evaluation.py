"""Evaluation script for ImageNet with ImageNet-X."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
from imagenet_x import FACTORS, plots
from imagenet_x.evaluate import ImageNetX, get_vanilla_transform
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from models.resnet import ResNet18, ResNet34, ResNet44, ResNet50, ResNet101


def main_process(args):
    """Main process."""

    # Load model.
    model_dict = {
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet44": ResNet44,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
    }
    assert args.arch in model_dict.keys(), "Model not supported"
    model = model_dict[args.arch](
        num_classes=1000,
        width=54 if args.rotations == 3 else 64,
        rotations=args.rotations,
        groupcosetmaxpool=args.groupcosetmaxpool,
    )
    summary(model, (2, 3, 224, 224), device="cpu")
    model = model.cuda()

    # Push model to GPU.
    model = DataParallel(model).cuda()

    # Load weights.
    checkpoint = torch.load(args.weights)
    r = model.load_state_dict(checkpoint["state_dict"])
    print(r)

    # Declare dataset
    transforms = get_vanilla_transform()
    dataset = ImageNetX(args.data, transform=transforms)

    # Get number of GPUs.
    ngpus_per_node = torch.cuda.device_count()

    # Evaluate model on ImageNetX using simple loop
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * ngpus_per_node,
        num_workers=args.workers,
        pin_memory=True,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target, annotations in tqdm(loader, desc="Evaluating on Imagenet-X"):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            mask = pred.eq(target.view_as(pred))
            correct += annotations[mask, :].to(dtype=torch.int).sum(dim=0)
            total += annotations.to(dtype=torch.int).sum(dim=0)

    # Compute accuracies per factor
    factor_accs = (correct / total).cpu().detach().numpy()  # type: ignore
    results = pd.DataFrame({"Factor": FACTORS, "acc": factor_accs}).sort_values(
        "acc", ascending=False
    )

    # Compute error ratios per factor
    results["Error ratio"] = (1 - results["acc"]) / (
        1 - (correct.sum() / total.sum()).item()  # type: ignore
    )

    # Plot results
    plots.plot_bar_plot(results, x="Factor", y="Error ratio")
    plt.savefig("imagenet_x.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("weights", type=str, help="path to model weights")
    parser.add_argument(
        "--data",
        default="/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/imagenet",
    )
    parser.add_argument("--workers", default=2, type=int, help="workers per GPU")
    parser.add_argument("--batch-size", default=64, type=int, help="batch size per gpu")

    # Input preprocessing settings.
    parser.add_argument("--grayscale", action="store_true", help="grayscale input")
    parser.add_argument("--nonorm", dest="normalize", action="store_false")

    # Architecture settings.
    parser.add_argument("--rotations", default=1, type=int, help="color rotations")
    parser.add_argument("--groupcosetmaxpool", action="store_true")
    parser.add_argument("--arch", default="resnet18", type=str)
    args = parser.parse_args()

    print(args)

    main_process(args)
