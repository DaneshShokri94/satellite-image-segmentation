# Satellite Image Segmentation

A comprehensive deep learning pipeline for semantic segmentation of satellite and aerial imagery using PyTorch.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Applications

- **Land Cover Classification** - Forest, water, urban, agriculture
- **Water Body Detection** - Rivers, lakes, floods, ice
- **Urban Mapping** - Buildings, roads, infrastructure
- **Vegetation Analysis** - Crop types, forest health
- **Change Detection** - Deforestation, urban expansion
- **Disaster Response** - Flood mapping, damage assessment
- **Ice/Snow Mapping** - Sea ice, glaciers, snow cover

## Features

- **9 Segmentation Architectures**: U-Net, U-Net++, DeepLabV3+, FPN, PSPNet, and more
- **40+ Encoder Backbones**: ResNet, EfficientNet, VGG, MobileNet, etc.
- **Multi-channel Support**: RGB, RGB+NIR, multispectral imagery
- **Binary & Multi-class**: Single class or multiple land cover types
- **Large Image Handling**: Automatic patch-based processing
- **Mixed Precision Training**: Faster training, less memory
- **Multiple Loss Functions**: Dice, Focal, Tversky, Lovasz, etc.

## Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Preparation](#-data-preparation)
- [Usage Examples](#-usage-examples)
- [Architectures & Encoders](#-architectures--encoders)
- [Hyperparameters](#-hyperparameters)
- [Multi-class Segmentation](#-multi-class-segmentation)

---

## 🔧 Installation

### Step 1: Create Conda Environment

```bash
conda create -n satellite-seg python=3.10 -y
conda activate satellite-seg
```

### Step 2: Install PyTorch

**CUDA 11.8 (NVIDIA GPU):**
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

**CUDA 12.1:**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**CPU Only:**
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

**macOS (Apple Silicon):**
```bash
conda install pytorch torchvision -c pytorch
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python check_installation.py
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

```
data/
├── train/          # Training images
├── train_masks/    # Training masks
├── val/            # Validation images
├── val_masks/      # Validation masks
├── test/           # Test images
└── test_masks/     # Test masks
```

### 2. Train Model

```bash
# Basic training
python train.py

# Custom configuration
python train.py --arch unet --encoder resnet50 --epochs 100 --batch-size 8
```

### 3. Run Inference

```bash
python inference.py --weights outputs/best_model/checkpoints/best_model.pth --input new_images/
```

---

## Data Preparation

### Directory Structure

```
data/
├── train/
│   ├── image_001.tif
│   ├── image_002.png
│   └── ...
├── train_masks/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── val/
│   └── ...
├── val_masks/
│   └── ...
├── test/
│   └── ...
└── test_masks/
    └── ...
```

### Supported Formats

- **Images**: TIFF, TIF, PNG, JPG, JPEG, BMP, WebP, GIF
- **Masks**: Same formats (PNG recommended)

### Mask Format

**Binary Segmentation (1 class):**
- Background: 0 (black)
- Target class: 255 or 1 (white)

**Multi-class Segmentation (N classes):**
- Class 0: pixel value 0
- Class 1: pixel value 1
- Class 2: pixel value 2
- ... and so on

---

## 📖 Usage Examples

### Example 1: Water Body Detection (Binary)

```bash
python train.py \
    --arch unet \
    --encoder resnet50 \
    --num-classes 1 \
    --epochs 100 \
    --batch-size 8 \
    --loss dice
```

### Example 2: Land Cover Classification (Multi-class)

```bash
python train.py \
    --arch deeplabv3plus \
    --encoder efficientnet-b4 \
    --num-classes 5 \
    --epochs 150 \
    --loss ce \
    --batch-size 8
```

### Example 3: 4-Band Imagery (RGB + NIR)

```bash
python train.py \
    --arch unet \
    --encoder resnet50 \
    --in-channels 4 \
    --num-classes 3 \
    --pretrained none
```

### Example 4: High Resolution (Large Images)

```bash
python train.py \
    --patch-size 512 \
    --stride 256 \
    --batch-size 4 \
    --arch fpn \
    --encoder resnet34
```

### Example 5: Limited GPU Memory

```bash
python train.py \
    --arch unet \
    --encoder mobilenet_v2 \
    --batch-size 4 \
    --patch-size 256 \
    --amp
```

### Example 6: Ice Detection (like SWOT data)

```bash
python train.py \
    --arch unetplusplus \
    --encoder resnet50 \
    --num-classes 1 \
    --loss focal \
    --focal-gamma 2.0 \
    --epochs 100
```

---

## Architectures & Encoders

### Segmentation Architectures

| Architecture | Description | Best For |
|--------------|-------------|----------|
| `unet` | Classic U-Net | General purpose, good baseline |
| `unetplusplus` | U-Net++ with nested connections | Fine details, boundaries |
| `deeplabv3plus` | Atrous convolutions + decoder | High accuracy, large objects |
| `fpn` | Feature Pyramid Network | Multi-scale objects |
| `pspnet` | Pyramid Pooling | Global context |
| `linknet` | Lightweight | Real-time, edge devices |
| `manet` | Multi-scale Attention | Variable object sizes |
| `pan` | Pyramid Attention | Attention-based |

### Encoder Backbones

| Family | Options | Notes |
|--------|---------|-------|
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101` | Best balance |
| EfficientNet | `efficientnet-b0` to `efficientnet-b7` | High accuracy |
| VGG | `vgg16`, `vgg19`, `vgg16_bn`, `vgg19_bn` | Classic |
| MobileNet | `mobilenet_v2` | Lightweight |
| DenseNet | `densenet121`, `densenet169`, `densenet201` | Dense connections |
| MiT | `mit_b0` to `mit_b5` | Transformer-based |

---

## ⚙️ Hyperparameters

### Loss Functions

```bash
# Dice Loss (default) - Good for imbalanced data
python train.py --loss dice

# Cross Entropy - Multi-class segmentation
python train.py --loss ce --num-classes 5

# Focal Loss - Hard examples, small objects
python train.py --loss focal --focal-gamma 2.0

# Tversky Loss - Control precision/recall
python train.py --loss tversky --tversky-alpha 0.7 --tversky-beta 0.3

# Jaccard/IoU Loss
python train.py --loss jaccard
```

### Optimizers

```bash
# AdamW (default)
python train.py --optimizer adamw --lr 0.0001

# SGD with momentum
python train.py --optimizer sgd --lr 0.01 --momentum 0.9

# Adam
python train.py --optimizer adam --lr 0.0001
```

### Learning Rate Schedulers

```bash
# ReduceLROnPlateau (default)
python train.py --scheduler plateau --scheduler-patience 5

# Cosine Annealing
python train.py --scheduler cosine

# Step Decay
python train.py --scheduler step --scheduler-step-size 30
```

---

## 🎨 Multi-class Segmentation

### Prepare Multi-class Masks

Masks should contain class indices (0, 1, 2, ..., N-1):

```python
# Example: 4-class land cover
# 0 = Background
# 1 = Water
# 2 = Vegetation
# 3 = Urban
```

### Train Multi-class Model

```bash
python train.py \
    --num-classes 4 \
    --loss ce \
    --arch deeplabv3plus \
    --encoder resnet50
```

### Color Mapping (for visualization)

```python
# Example color map
CLASS_COLORS = {
    0: (0, 0, 0),       # Background - Black
    1: (0, 0, 255),     # Water - Blue
    2: (0, 255, 0),     # Vegetation - Green
    3: (255, 0, 0),     # Urban - Red
}
```

---

## 📊 Output Structure

```
outputs/unet_resnet50_20240101_120000/
├── config.json           # Training configuration
├── history.json          # Loss and metrics history
├── checkpoints/
│   ├── best_model.pth    # Best model weights
│   └── checkpoint_epoch_*.pth
└── predictions/
    ├── image_001.png
    └── ...
```

---

## 🛠️ All Command Line Options

```bash
python train.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | unet | Architecture (unet, deeplabv3plus, etc.) |
| `--encoder` | resnet50 | Encoder backbone |
| `--in-channels` | 3 | Input channels (3=RGB, 4=RGBNIR) |
| `--num-classes` | 1 | Output classes (1=binary) |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--loss` | dice | Loss function |
| `--optimizer` | adamw | Optimizer |
| `--scheduler` | plateau | LR scheduler |
| `--patch-size` | 256 | Patch size |
| `--stride` | 128 | Stride for patches |
| `--amp` | True | Mixed precision |

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
python train.py --batch-size 4 --patch-size 256
```

### Slow Training

```bash
python train.py --encoder resnet34 --num-workers 8
```

### Poor Results

```bash
# Try different loss
python train.py --loss focal

# More epochs
python train.py --epochs 200

# Different architecture
python train.py --arch deeplabv3plus
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch](https://pytorch.org/)
