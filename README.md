# CEConv - Color Equivariant Convolutional Networks

[[ArXiv](https://arxiv.org/abs/2310.19368)] - NeurIPS 2023, by Attila Lengyel, Ombretta Strafforello, Robert-Jan Bruintjes, Alexander Gielisse, and Jan van Gemert.

## Abstract
Color is a crucial visual cue readily exploited by Convolutional Neural Networks (CNNs) for object recognition. However, CNNs struggle if there is data imbalance between color variations introduced by accidental recording conditions. Color invariance addresses this issue but does so at the cost of removing all color information, which sacrifices discriminative power. In this paper, we propose Color Equivariant Convolutions (CEConvs), a novel deep learning building block that enables shape feature sharing across the color spectrum while retaining important color information. We extend the notion of equivariance from geometric to photometric transformations by incorporating parameter sharing over hue-shifts in a neural network. We demonstrate the benefits of CEConvs in terms of downstream performance to various tasks and improved robustness to color changes, including train-test distribution shifts. Our approach can be seamlessly integrated into existing architectures, such as ResNets, and offers a promising solution for addressing color-based domain shifts in CNNs.

## Setup

Create a local clone of this repository:
```bash
git clone https://github.com/Attila94/CEConv
```

See `requirements.txt` for the required Python packages. You can install them using:
```bash
pip install -r requirements.txt
```

Install CEConv:
```bash
python setup.py install
```

Set the required environment variables:
```bash
export WANDB_DIR = /path/to/wandb/directory  # Store wandb logs here.
export DATA_DIR = /path/to/data/directory  # Store datasets here.
```

## How to use

CEConv can be used in the same way as a regular Conv2d layer. The following code snippet shows how to use CEConv in a CNN architecture:

```python
import torch
import torch.nn as nn
from ceconv import CEConv2d

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Args: input rotations, output rotations, input channels, output channels, kernel size, padding.
        self.conv1 = CEConv2d(1, 3, 3, 32, 3, padding=1)
        self.conv2 = CEConv2d(3, 3, 32, 64, 3, padding=1)
        
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Average pooling over spatial and color dimensions.
        x = torch.mean(x, dim=(2, 3, 4))
        
        x = self.fc(x)
        return x
```

## Experiments

The experiments from the paper can be reproduced by running the following commands.

### ColorMNIST
**Generate ColorMNIST datasets**
```bash
python -m experiments.color_mnist.colormnist_longtailed
python -m experiments.color_mnist.colormnist_biased --std 0
python -m experiments.color_mnist.colormnist_biased --std 12
python -m experiments.color_mnist.colormnist_biased --std 24
python -m experiments.color_mnist.colormnist_biased --std 36
python -m experiments.color_mnist.colormnist_biased --std 48
python -m experiments.color_mnist.colormnist_biased --std 60
python -m experiments.color_mnist.colormnist_biased --std 1000000
```

**Longtailed ColorMNIST**
```bash
# Baseline, grayscale and color jitter.
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20 --grayscale 
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20 --jitter 0.5

# Color Equivariant with and without group coset pooling and color jitter.
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --jitter 0.5
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --groupcosetpool

# Hybrid Color Equivariant architectures.
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 19 --ce_layers 2 --separable --groupcosetpool
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 18 --ce_layers 4 --separable --groupcosetpool
```

**Biased ColorMNIST**
```bash
# Baseline and grayscale.
python -m experiments.color_mnist.train_biased --std $2 --rotations 1 --planes 20 
python -m experiments.color_mnist.train_biased --std $2 --rotations 1 --planes 20 --grayscale

# Color equivariant with and without group coset pooling.
python -m experiments.color_mnist.train_biased --std $2 --rotations 3 --planes 17 --separable
python -m experiments.color_mnist.train_biased --std $2 --rotations 3 --planes 17 --separable --groupcosetpool

# Hybrid Color Equivariant architectures.
python -m experiments.color_mnist.train_biased --std $2 --rotations 3 --planes 19 --ce_layers 2 --separable --groupcosetpool
python -m experiments.color_mnist.train_biased --std $2 --rotations 3 --planes 18 --ce_layers 4 --separable --groupcosetpool
```

### Classification
Only the commands for CIFAR10 are provided. The commands for other datasets are similar, where the dataset can be specified using the `--dataset 'cifar100'`. Optionally the `--architecture 'resnet18'` flag can be added to use a ResNet18 architecture instead of a ResNet44.

```bash
# Baseline, grayscale and color jitter.
python -m experiments.classification.train --rotations 1
python -m experiments.classification.train --rotations 1 --grayscale
python -m experiments.classification.train --rotations 1 --jitter 0.5

# Color Equivariant with and without group coset pooling and color jitter.
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --jitter 0.5

# Hybrid Color Equivariant architectures.
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 1 --width 31
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 2 --width 30
```

### ImageNet
The ImageNet training script uses the [NVIDIA DALI](https://github.com/NVIDIA/DALI) library for accelerated data loading.

```bash
# Baseline.
python -m experiments.imagenet.main --rotations 1 --jitter 0.0 --arch 'resnet18'

# Color Equivariant.
python -m experiments.imagenet.main --rotations 3 --batch-size 256 --jitter 0.0 --workers 4 --arch 'resnet18' --groupcosetmaxpool --separable

# Hybrid Color Equivariant architectures.
python -m experiments.imagenet.main --rotations 3 --batch-size 256 --jitter 0.0 --workers 4 --arch 'resnet18' --network_width 63 --run_name 'hybrid_1' --groupcosetmaxpool --separable --ce_stages 1
python -m experiments.imagenet.main --rotations 3 --batch-size 256 --jitter 0.0 --workers 4 --arch 'resnet18' --network_width 63 --run_name 'hybrid_2' --groupcosetmaxpool --separable --ce_stages 2
python -m experiments.imagenet.main --rotations 3 --batch-size 256 --jitter 0.0 --workers 4 --arch 'resnet18' --network_width 61 --run_name 'hybrid_3' --groupcosetmaxpool --separable --ce_stages 3
```


## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{
    lengyel2023color,
    title={Color Equivariant Convolutional Networks},
    author={Attila Lengyel and Ombretta Strafforello and Robert-Jan Bruintjes and Alexander Gielisse and Jan van Gemert},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=xz8j3r3oUA}
}
```
