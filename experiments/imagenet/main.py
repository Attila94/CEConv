import argparse
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from experiments.imagenet.imagenet_tfrecord import ImageNet_TFRecord
from models.resnet import *
from models.resnet_hybrid import *
from torch import distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
from torchvision.transforms.functional import adjust_hue


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


def main_process(args):
    # Initialize wandb.
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    # Run name.
    run_name = "{}_{}-{}-jitter_{}".format(
        args.arch,
        args.rotations,
        str(args.groupcosetmaxpool).lower(),
        str(args.jitter).replace(".", "_"),
    )
    if args.run_name != "":
        run_name += "-" + args.run_name
    args.run_name = run_name
    if args.grayscale:
        args.run_name += "-grayscale"
    if args.normalize == False:
        args.run_name += "-no_norm"
    if not args.separable:
        args.run_name += "-nonsep"

    # Set address for master process to localhost since we use a single node.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12359 + np.random.randint(0, 1000))

    # Use all gpus pytorch can find.
    args.world_size = torch.cuda.device_count()
    print("Found {} GPUs:".format(args.world_size))
    for i in range(args.world_size):
        print("{} : {}".format(i, torch.cuda.get_device_name(i)))

    # Total batch size = batch size per gpu * ngpus.
    args.total_batch_size = args.world_size * args.batch_size

    print("\nCUDNN VERSION: {}\n".format(cudnn.version()))
    cudnn.benchmark = True
    assert cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if not len(args.data):
        raise Exception("error: No data set provided")

    # Start processes for all gpus.
    spawn(gpu_process, nprocs=args.world_size, args=(args,))


def gpu_process(gpu, args):
    out_dir = os.path.join(os.environ["OUT_DIR"], "color_equivariance/imagenet")
    if gpu == 0 and not args.debug:
        run_id = args.run_id if args.run_id != "" else None
        wandb.init(
            project="color-equivariance-imagenet",
            entity="tudcv",
            config=vars(args),
            id=run_id,
            resume=run_id is not None,
            name=args.run_name,
            dir=os.environ["WANDB_DIR"],
            mode="offline" if args.offline else "online",
        )
        os.makedirs(out_dir, exist_ok=True)

    # Each gpu runs in a separate process.
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=gpu, world_size=args.world_size
    )

    # Create model.
    if args.ce_stages is not None:
        model_dict = {
            "resnet18": HybridResNet18,
            "resnet34": HybridResNet34,
            "resnet44": HybridResNet44,
            "resnet50": HybridResNet50,
            "resnet101": HybridResNet101,
            "resnet152": HybridResNet152,
        }
    else:
        model_dict = {
            "resnet18": ResNet18,
            "resnet34": ResNet34,
            "resnet44": ResNet44,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "resnet152": ResNet152,
        }
    kwargs = {
        "rotations": args.rotations,
        "groupcosetmaxpool": args.groupcosetmaxpool,
        "separable": args.separable,
        "width": args.network_width,
        "num_classes": 1000,
        "ce_stages": args.ce_stages,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    assert args.arch in model_dict.keys(), "Model not supported"
    model = model_dict[args.arch](**kwargs)

    if gpu == 0:
        summary(model, (2, 3, 224, 224), device="cpu")

    # Set cudnn to deterministic setting.
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(gpu)
        torch.set_printoptions(precision=10)

    # Push model to gpu.
    model = model.cuda(gpu)

    # Scale learning rate based on global batch size.
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.0
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)
    else:
        raise ValueError("Optimizer not supported")

    # Use DistributedDataParallel for distributed training.
    model = DDP(model, device_ids=[gpu], output_device=gpu)

    # Define loss function (criterion) and optimizer.
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    best_prec1 = 0

    # Optionally resume from a checkpoint.
    if args.resume or args.torchhub:
        # Use a local scope to avoid dangling references.
        def resume():
            if args.torchhub:
                print("=> downloading checkpoint '{}'".format(args.torchhub))
                args.resume = "/tmp/checkpoint.pth.tar"
                torch.hub.download_url_to_file(args.torchhub, args.resume)
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(gpu)
                )
                # Remap torchhub ckpt to own naming
                new_ckpt = {}
                for k, v in checkpoint.items():
                    k = "module." + k
                    k = k.replace("layer1", "layers.0")
                    k = k.replace("layer2", "layers.1")
                    k = k.replace("layer3", "layers.2")
                    k = k.replace("layer4", "layers.3")
                    k = k.replace("downsample", "shortcut")
                    k = k.replace("fc", "linear")
                    new_ckpt[k] = v
                checkpoint = new_ckpt

            elif os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(gpu)
                )

            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                return 0

            epoch_loaded = None
            best_prec1 = 0.0
            if "state_dict" in checkpoint:
                args.start_epoch = checkpoint["epoch"]
                epoch_loaded = checkpoint["epoch"]
                best_prec1 = checkpoint["best_prec1"]
                model.load_state_dict(checkpoint["state_dict"])
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as e:
                    print(e)
                    print("Warning! Optimizer could not be loaded.")
            else:
                model.load_state_dict(checkpoint)
                print("Warning! Only weights have been loaded.")

            print(
                "=> loaded checkpoint '{}' (epoch {})".format(args.resume, epoch_loaded)
            )
            return best_prec1

        best_prec1 = resume()

    # Data loading code.
    train_loader = ImageNet_TFRecord(
        args.data,
        "train",
        args.batch_size,
        args.workers,
        gpu,
        args.world_size,
        is_training=True,
        jitter=args.jitter,
        grayscale=args.grayscale,
        subset=args.subset,
    )
    val_loader = ImageNet_TFRecord(
        args.data,
        "val",
        args.batch_size,
        args.workers,
        gpu,
        args.world_size,
        is_training=False,
        grayscale=args.grayscale,
    )

    # Only evaluate model, no training.
    if args.evaluate:
        [prec1, prec5, sup_prec1] = validate(val_loader, model, criterion, gpu, args)
        wandb.log({"val/top1": prec1, "val/top5": prec5})
        wandb.log({f"val/top1-{s}": v for (s, v) in sup_prec1.items()})
        return

    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch.
        train_time = train(train_loader, model, criterion, optimizer, epoch, gpu, args)
        total_time.update(train_time)

        # Evaluate on validation set.
        [prec1, prec5, sup_prec1] = validate(val_loader, model, criterion, gpu, args)

        if gpu == 0 and not args.debug:
            wandb.log({"val/top1": prec1, "val/top5": prec5})
            wandb.log({f"val/top1-{s}": v for (s, v) in sup_prec1.items()})

        # Remember best prec@1 and save checkpoint.
        if gpu == 0:
            filename = os.path.join(out_dir, args.run_name + ".pth.tar")
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                filename=filename,
            )
            if epoch == args.epochs - 1:
                print(
                    "##Top-1 {0}\n"
                    "##Top-5 {1}\n"
                    "##Perf  {2}".format(
                        prec1, prec5, args.total_batch_size / total_time.avg
                    )
                )

        train_loader.reset()
        val_loader.reset()

        # If in debug mode quit after 1st epoch.
        if args.debug and epoch == 1:
            break

    test_hue_shift(val_loader, model, gpu, args)


def train(train_loader, model, criterion, optimizer, epoch, gpu, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to train mode.
    model.train()

    train_loader_len = int(math.ceil(train_loader._size / args.batch_size))

    end = time.time()
    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda(gpu).long()

        # Lr schedule.
        adjust_learning_rate(
            args.lr, optimizer, epoch, i, train_loader_len, args.epochs
        )

        # If in debug mode, quit after 100 iterations.
        if args.debug:
            train_loader_len = 100
            if i > train_loader_len:
                break

        # Normalize input.
        # Use grayscale normalization for grayscale images or when rotations > 1.
        if args.normalize:
            input = normalize(input, grayscale=args.grayscale or args.rotations > 1)

        # Compute output.
        output = model(input)
        loss = criterion(output, target)

        # Compute gradient and do SGD step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy.
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging.
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)

            # to_python_float incurs a host<->device sync.
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if gpu == 0:  # Only print for main process.
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Speed {3:.3f} ({4:.3f})\t"
                    "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        epoch,
                        i,
                        train_loader_len,
                        args.world_size * args.batch_size / batch_time.val,
                        args.world_size * args.batch_size / batch_time.avg,
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

                # Do not commit to wandb at last iteration so we can add
                # the test stats to the same step.
                if not args.debug:
                    wandb.log(
                        {
                            "train/loss": losses.val,
                            "train/top1": top1.val,
                            "train/top5": top5.val,
                        },
                        step=epoch * train_loader_len + i,
                        commit=i + 1 < train_loader_len,
                    )

    return batch_time.avg


def validate(val_loader, model, criterion, gpu, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    supercategories = [
        ("animal", (0, 397)),
        ("nature", (970, 998)),
        ("manmade", (398, 922)),
        ("manmade", (999, 999)),
        ("food", (923, 969)),
    ]
    sup_accs = {
        "animal": AverageMeter(),
        "nature": AverageMeter(),
        "manmade": AverageMeter(),
        "food": AverageMeter(),
    }

    if args.classwise:
        class_accs = {f"class{c:04d}": AverageMeter() for c in range(1000)}

    # switch to evaluate mode
    model.eval()

    val_loader_len = int(val_loader._size / args.batch_size)

    end = time.time()
    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda(gpu).long()

        # Normalize input.
        # Use grayscale normalization for grayscale images or when rotations > 1.
        if args.normalize:
            input = normalize(input, grayscale=args.grayscale or args.rotations > 1)

        # Compute output.
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # Measure accuracy and record loss.
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss.data, args.world_size)
        prec1 = reduce_tensor(prec1, args.world_size)
        prec5 = reduce_tensor(prec5, args.world_size)
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # Measure supercatorgory accuracy per sample by checking if the target matches the range given
        # for the supercategory.
        for j in range(input.size(0)):
            for s, (start, end) in supercategories:
                if start <= target[j] <= end:
                    if output[j].argmax().item() == target[j]:
                        sup_accs[s].update(1.0, 1)
                    else:
                        sup_accs[s].update(0.0, 1)

        if args.classwise:
            for j in range(input.size(0)):
                if output[j].argmax().item() == target[j]:
                    class_accs[f"class{target[j]:04d}"].update(1.0, 1)
                else:
                    class_accs[f"class{target[j]:04d}"].update(0.0, 1)

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        if gpu == 0 and i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Speed {2:.3f} ({3:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    val_loader_len,
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )


        if args.debug and i == 10:
            break

    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    print("Supercategory accuracies:")
    for s, acc in sup_accs.items():
        print(f"Prec@1-{s}: {acc.val} ({acc.avg:.3f})")
    if args.classwise:
        print("Class accuracies:")
        accs = sorted(list([(c, acc.avg) for (c, acc) in class_accs.items()]), key=lambda x: x[0])
        print(accs)

    supcat_items = list(sup_accs.items())
    if args.classwise:
        supcat_items = supcat_items + list(class_accs.items())
    return [top1.avg, top5.avg, {s: v.avg for (s, v) in supcat_items}]


def normalize(batch: torch.Tensor, grayscale: bool) -> torch.Tensor:
    """Normalize batch of images.

    Args:
        batch: Batch of images to normalize.
        grayscale: Whether the images are grayscale or not.
        undo: Whether to undo the normalization or not.
    """

    if not grayscale:
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.485], device=batch.device).view(1, 1, 1, 1)
        std = torch.tensor([0.229], device=batch.device).view(1, 1, 1, 1)
    return (batch - mean) / std


def test_hue_shift(val_loader, model, gpu, args):
    # Switch to evaluate mode.
    model.eval()

    test_jitter = np.linspace(-0.5, 0.5, 37)
    top_1_dict = {}
    top_5_dict = {}
    for i in test_jitter:
        top_1_dict["{:.4f}".format(i)] = AverageMeter()
        top_5_dict["{:.4f}".format(i)] = AverageMeter()

    val_loader_len = int(val_loader._size / args.batch_size)
    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda(gpu).long()

        for j in test_jitter:
            # Apply hue shift.
            x = adjust_hue(input, j)

            # Normalize images.
            if args.normalize:
                x = normalize(x, grayscale=args.grayscale or args.rotations > 1)

            # Forward pass and compute loss.
            with torch.no_grad():
                y_pred = model(x)

            # Logging.
            prec1, prec5 = accuracy(y_pred, target, topk=(1, 5))
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)
            top_1_dict["{:.4f}".format(j)].update(to_python_float(prec1), input.size(0))
            top_5_dict["{:.4f}".format(j)].update(to_python_float(prec5), input.size(0))

        if gpu == 0 and i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    val_loader_len,
                    top1=top_1_dict["0.0000"],
                    top5=top_5_dict["0.0000"],
                )
            )

        if args.debug and i == 10:
            break

    for i in test_jitter:
        print(
            " * Hue shift {0:.4f}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(
                i,
                top1=top_1_dict["{:.4f}".format(i)],
                top5=top_5_dict["{:.4f}".format(i)],
            )
        )

    # Log metrics, and reset metrics.
    if gpu == 0 and not args.debug:
        columns = ["hue", "top1", "top5"]
        test_table = wandb.Table(columns=columns)

        for i in test_jitter:
            test_table.add_data(
                i,
                top_1_dict["{:.4f}".format(i)].avg,
                top_5_dict["{:.4f}".format(i)].avg,
            )
            top_1_dict["{:.4f}".format(i)].reset()
            top_5_dict["{:.4f}".format(i)].reset()

        # Log test table with wandb.
        wandb.log({"test_table": test_table})  # type: ignore


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8] + "_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, optimizer, epoch, step, len_epoch, num_epochs):
    """
    LR schedule that should yield 76% converged accuracy with batch size 256.

    Args:
        lr: Initial learning rate.
        optimizer: Optimizer for which to adjust the learning rate.
        epoch: Current epoch.
        step: Current step in the current epoch.
        len_epoch: Number of steps in the current epoch.
        num_epochs: Total number of epochs.
    """

    scaling = num_epochs / 90
    factor = epoch // (30 * scaling)

    if epoch >= 80 * scaling:
        factor = factor + 1

    lr = lr * (0.1**factor)

    """Warmup"""
    if epoch < 5 * scaling:
        lr = lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch * scaling)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--data",
        help="path(s) to dataset",
        default="/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/imagenet/tfrecords",
    )
    parser.add_argument(
        "--subset", default=1.0, type=float, help="subset of data to use"
    )
    parser.add_argument(
        "--workers", default=2, type=int, help="number of data loading workers per GPU"
    )
    parser.add_argument("--run_name", default="", type=str, help="name of the run")
    parser.add_argument(
        "--epochs", default=90, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--optimizer", default="sgd", type=str, help="optimizer to use")
    parser.add_argument(
        "--start-epoch", default=0, type=int, help="manual epoch number"
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, help="batch size per gpu (default: 64)"
    )
    parser.add_argument(
        "--lr", default=0.1, type=float, help="init. learning rate for bs=256."
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd", default=1e-4, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--print-freq", default=10, type=int, metavar="N", help="print frequency"
    )
    parser.add_argument(
        "--resume", default="", type=str, help="path to latest checkpoint"
    )
    parser.add_argument(
        "--torchhub", default="", type=str, help="Torch Hub path to checkpoint"
    )
    parser.add_argument(
        "--run_id", default="", type=str, help="wandb run id to resume from."
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate model on validation set"
    )
    parser.add_argument(
        "--dali_cpu", action="store_true", help="Runs CPU based version of DALI."
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Run short debug script.")
    parser.add_argument("--offline", action="store_true", help="Don't log to W&B.")

    # Input preprocessing settings.
    parser.add_argument(
        "--jitter", default=0.0, type=float, help="jitter strength (default: 0.0)"
    )
    parser.add_argument(
        "--grayscale", action="store_true", help="grayscale input (default: False)"
    )
    parser.add_argument(
        "--nonorm", dest="normalize", action="store_false", help="no input norm."
    )

    # Architecture settings.
    parser.add_argument(
        "--rotations", default=1, type=int, help="number of color rotations in group"
    )
    parser.add_argument(
        "--groupcosetmaxpool", action="store_true", help="use coset max pooling"
    )
    parser.add_argument(
        "--arch", default="resnet18", type=str, help="Network architecture."
    )
    parser.add_argument(
        "--network_width", default=None, type=int, help="Network width."
    )
    parser.add_argument(
        "--separable", action="store_true", help="use separable convolutions"
    )
    parser.add_argument(
        "--ce_stages", default=None, type=int, help="number of equivariant stages"
    )

    # Validation settings
    parser.add_argument(
        "--classwise", action="store_true", help="report classwise accuracy",
    )

    args = parser.parse_args()

    print(args)

    main_process(args)
