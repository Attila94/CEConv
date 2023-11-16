import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import wandb

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
from torchinfo import summary
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset, Subset

from models.cnn import CECNN, CNN


class PL_model(pl.LightningModule):
    def __init__(self, args):
        super(PL_model, self).__init__()

        # Logging.
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.preds = torch.Tensor()  # log for confmat
        self.gts = torch.Tensor()  # log for confmat

        # Model definition.
        if args.rotations == 1:
            self.model = CNN(args.planes)
        elif args.rotations > 1:
            self.model = CECNN(
                args.planes,
                args.rotations,
                groupcosetmaxpool=args.groupcosetpool,
                separable=args.separable,
                ce_layers=args.ce_layers,
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--planes", type=int, default=20, help="channels in CNN")
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        parser.add_argument("--rotations", type=int, default=1, help="CECNN rotations")
        parser.add_argument("--groupcosetpool", action="store_true", help="cosetpool")
        parser.add_argument("--separable", action="store_true", help="separable CEConv")
        parser.add_argument("--ce_layers", type=int, default=7, help="CECNN layers")
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

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass and compute loss.
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)

        # Logging.
        batch_acc = self.train_acc(y_pred, y)
        self.log("train_acc_step", batch_acc)
        self.log("train_loss_step", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log("train_acc_epoch", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass and compute loss.
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)

        # Logging.
        self.test_acc.update(y_pred.detach().cpu(), y.cpu())

        # Log confusion matrix.
        if self.current_epoch == args.epochs - 1:
            self.preds = torch.cat((self.preds, F.softmax(y_pred, 1).detach().cpu()), 0)
            self.gts = torch.cat((self.gts, y.cpu()), 0)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        self.log("test_acc_epoch", self.test_acc.compute())
        self.test_acc.reset()

        # Only log confusion matrix for last epoch.
        if self.current_epoch == args.epochs - 1:
            classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            self.logger.experiment.log(  # type: ignore
                {
                    "test_conf_mat": wandb.plot.confusion_matrix(  # type: ignore
                        probs=self.preds.numpy(),
                        y_true=self.gts.numpy(),
                        class_names=classnames,
                    )
                }
            )

            self.preds = torch.Tensor()
            self.gts = torch.Tensor()


class CustomDataset(TensorDataset):
    def __init__(self, dataset, jitter=0.0, grayscale=False):
        self.tensors = dataset.tensors
        self.jitter = jitter

        # Convert to grayscale.
        if grayscale:
            self.tensors = (
                torch.mean(self.tensors[0], dim=1, keepdim=True).repeat(1, 3, 1, 1),
                self.tensors[1],
            )

    def __getitem__(self, index):
        x, y = tuple(tensor[index] for tensor in self.tensors)
        if self.jitter != 0.0:
            x = TF.adjust_hue(x, np.random.uniform(-self.jitter, self.jitter))
        return x, y


def getDataset(std=0, subset_samples=None):
    # Load train dataset files.
    train = CustomDataset(
        torch.load(
            os.environ["DATA_DIR"] + "/colormnist_biased/train_{}.pt".format(std)
        ),
        jitter=args.jitter,
        grayscale=args.grayscale,
    )
    test = CustomDataset(
        torch.load(
            os.environ["DATA_DIR"] + "/colormnist_biased/test_{}.pt".format(std)
        ),
        grayscale=args.grayscale,
    )

    # Use only a subset of train set.
    if subset_samples is not None:
        train = Subset(train, np.random.choice(len(train), subset_samples))

    # Data loaders.
    trainloader = DataLoader(
        train,
        batch_size=args.bs,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    testloader = DataLoader(
        test,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    # Print dataset sizes.
    print(
        "train:",
        torch.unique(train.dataset.tensors[1][train.indices], return_counts=True),  # type: ignore
    )
    print("test:", torch.unique(test.tensors[1], return_counts=True))

    return trainloader, testloader


def main(args) -> None:
    # Create temp dir for wandb.
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    # Use fixed seed.
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Get data loaders.
    trainloader, testloader = getDataset(
        std=args.std, subset_samples=args.subset_samples
    )
    args.steps_per_epoch = len(trainloader)

    # Initialize model.
    model = PL_model(args)
    summary(model.model, (2, 3, 28, 28))

    # Callbacks and loggers.
    run_name = "std_{}-subset_{}-seed_{}-rotations_{}".format(
        args.std, args.subset_samples, args.seed, args.rotations
    )
    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="ceconv-colormnist-new",
        entity="tudcv",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Define callback to store model weights.
    weights_dir = os.path.join(
        os.environ["OUT_DIR"], "color_equivariance/colormnist_biased/"
    )
    os.makedirs(weights_dir, exist_ok=True)
    weights_name = run_name + ".pth.tar"
    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_dir,
        filename=weights_name,
        every_n_epochs=args.epochs,
        save_weights_only=True,
    )

    # Train model.
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=mylogger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=args.epochs,
        log_every_n_steps=40,
        deterministic=(args.seed is not None),
        check_val_every_n_epoch=1
        if args.subset_samples is None
        else 100000 // args.subset_samples,
    )
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=testloader,
    )


if __name__ == "__main__":

    # Training settings.
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=250, help="batch size")
    parser.add_argument("--test-bs", type=int, default=250, help="test batch size")
    parser.add_argument("--std", type=int, default=0, help="dataset std")
    parser.add_argument("--grayscale", action="store_true", help="use grayscale")
    parser.add_argument("--jitter", type=float, default=0.0, help="jitter")
    parser.add_argument("--subset-samples", type=int, default=1000, help="use subset")
    parser.add_argument("--epochs", type=int, default=1500, help="number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser = PL_model.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)
