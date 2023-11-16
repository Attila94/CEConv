"""Image classification experiments for Color Equivariant Convolutional Networks."""

import argparse
import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchinfo import summary
from torchvision.transforms.functional import adjust_hue

from experiments.classification.datasets import get_dataset, normalize
from models.resnet import ResNet18, ResNet44
from models.resnet_hybrid import HybridResNet18, HybridResNet44


class PL_model(pl.LightningModule):
    def __init__(self, args) -> None:
        super(PL_model, self).__init__()

        # Logging.
        self.save_hyperparameters()

        # Store predictions and ground truth for computing confusion matrix.
        self.preds = torch.tensor([])
        self.gts = torch.tensor([])

        # Store accuracy metrics for logging.
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        # Store accuracy metrics for testing.
        self.test_acc_dict = {}
        self.test_jitter = np.linspace(-0.5, 0.5, 37)
        for i in self.test_jitter:
            self.test_acc_dict["test_acc_{:.4f}".format(i)] = torchmetrics.Accuracy()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Model definition.
        if args.ce_stages is not None:
            architectures = {"resnet18": HybridResNet18, "resnet44": HybridResNet44}
        else:
            architectures = {"resnet18": ResNet18, "resnet44": ResNet44}
        assert args.architecture in architectures.keys(), "Model not supported."
        kwargs = {
            "rotations": args.rotations,
            "groupcosetmaxpool": args.groupcosetmaxpool,
            "separable": args.separable,
            "width": args.width,
            "num_classes": len(args.classes),
            "ce_stages": args.ce_stages,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = architectures[args.architecture](**kwargs)

        # Print model summary.
        resolution = 32 if args.architecture == "resnet44" else 224
        summary(self.model, (2, 3, resolution, resolution), device="cpu")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        parser.add_argument("--groupcosetmaxpool", action="store_true")
        parser.add_argument("--architecture", default="resnet44", type=str)
        parser.add_argument("--rotations", type=int, default=1, help="no. hue rot.")
        parser.add_argument("--separable", action="store_true", help="separable conv")
        parser.add_argument("--width", type=int, default=None, help="network width")
        parser.add_argument("--ce_stages", type=int, default=None, help="ce res stages")
        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.wd
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
                optimizer,
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
            ),
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        x, y = batch

        # Normalize images.
        if args.normalize:
            x = normalize(x, grayscale=args.grayscale or args.rotations > 1)

        # Forward pass and compute loss.
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        # Logging.
        batch_acc = self.train_acc(y_pred, y)
        self.log("train_acc_step", batch_acc)
        self.log("train_loss_step", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        self.log("train_acc_epoch", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        x, y = batch

        # Normalize images.
        if args.normalize:
            x = normalize(x, grayscale=args.grayscale or args.rotations > 1)

        # Forward pass and compute loss.
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        # Logging.
        self.test_acc.update(y_pred.detach().cpu(), y.cpu())

        return {"loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        self.log("test_acc_epoch", self.test_acc.compute())
        self.test_acc.reset()

    def test_step(self, batch, batch_idx) -> None:
        x_org, y = batch

        for i in self.test_jitter:
            # Apply hue shift.
            x = adjust_hue(x_org, i)

            # Normalize images.
            if args.normalize:
                x = normalize(x, grayscale=args.grayscale or args.rotations > 1)

            # Forward pass and compute loss.
            y_pred = self.model(x)

            # Logging.
            self.test_acc_dict["test_acc_{:.4f}".format(i)].update(
                y_pred.detach().cpu(), y.cpu()
            )

            # If no hue shift, log predictions and ground truth.
            if int(i) == 0:
                self.preds = torch.cat(
                    (self.preds, F.softmax(y_pred, 1).detach().cpu()), 0
                )
                self.gts = torch.cat((self.gts, y.cpu()), 0)

    def test_epoch_end(self, outputs):
        # Log metrics and predictions, and reset metrics.
        columns = ["hue", "acc"]
        test_table = wandb.Table(columns=columns)

        for i in self.test_jitter:
            test_table.add_data(
                i, self.test_acc_dict["test_acc_{:.4f}".format(i)].compute().item()
            )
            self.test_acc_dict["test_acc_{:.4f}".format(i)].reset()

        # Log test table with wandb.
        self.logger.experiment.log({"test_table": test_table})  # type: ignore

        # Log confusion matrix with wandb.
        self.logger.experiment.log(  # type: ignore
            {
                "test_conf_mat": wandb.plot.confusion_matrix(  # type: ignore
                    probs=self.preds.numpy(),
                    y_true=self.gts.numpy(),
                    class_names=args.classes,
                )
            }
        )


def main(args) -> None:
    # Create temp dir for wandb.
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    # Use fixed seed.
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Get data loaders.
    trainloader, testloader = get_dataset(args)
    args.steps_per_epoch = len(trainloader)
    args.epochs = math.ceil(args.epochs / args.split)

    # Initialize model.
    model = PL_model(args)

    # Callbacks and loggers.
    run_name = "{}-{}_{}-{}-jitter_{}-split_{}-seed_{}".format(
        args.dataset,
        args.architecture,
        args.rotations,
        str(args.groupcosetmaxpool).lower(),
        str(args.jitter).replace(".", "_"),
        str(args.split).replace(".", "_"),
        args.seed,
    )
    if args.grayscale:
        run_name += "-grayscale"
    if not args.normalize:
        run_name += "-no_norm"
    if args.ce_stages is not None:
        run_name += "-{}_stages".format(args.ce_stages)
    if args.run_name is not None:
        run_name += "-" + args.run_name
    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="color-equivariance-classification",
        entity="tudcv",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Define callback to store model weights.
    weights_dir = os.path.join(
        os.environ["OUT_DIR"], "color_equivariance/classification/"
    )
    os.makedirs(weights_dir, exist_ok=True)
    weights_name = run_name + ".pth.tar"
    checkpoint_callback = ModelCheckpoint(dirpath=weights_dir, filename=weights_name)

    # Train model.
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=mylogger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=args.epochs,
        log_every_n_steps=10,
        deterministic=(args.seed is not None),
        check_val_every_n_epoch=20,
    )

    # Get path to latest model weights if they exist.
    if args.resume:
        checkpoint_files = os.listdir(weights_dir)
        weights_path = [
            os.path.join(weights_dir, f) for f in checkpoint_files if weights_name in f
        ]
        weights_path = weights_path[0] if len(weights_path) > 0 else None
    else:
        weights_path = None

    # Train model.
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=[testloader],
        ckpt_path=weights_path,
    )

    # Test model.
    trainer.test(model, dataloaders=testloader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Dataset settings.
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument(
        "--split", default=1.0, type=float, help="Fraction of training set to use."
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument(
        "--jitter", type=float, default=0.0, help="color jitter strength"
    )
    parser.add_argument(
        "--nonorm", dest="normalize", action="store_false", help="no input norm."
    )

    # Training settings.
    parser.add_argument(
        "--bs", type=int, default=256, help="training batch size (default: 256)"
    )
    parser.add_argument(
        "--test-bs", type=int, default=256, help="test batch size (default: 256)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of epochs (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="name of run (default: None)"
    )
    parser.add_argument(
        "--resume", dest="resume", action="store_true", help="resume training."
    )

    parser = PL_model.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)
