#!/usr/bin/env python3
"""
Satellite Image Segmentation
=============================
A comprehensive deep learning pipeline for semantic segmentation of satellite 
and aerial imagery using PyTorch and Segmentation Models PyTorch (SMP).

Supports:
- Binary segmentation (e.g., water vs land)
- Multi-class segmentation (e.g., land cover classification)
- Any satellite/aerial imagery (Sentinel, Landsat, SPOT, drone, etc.)

Usage:
    python train.py --encoder resnet50 --arch unet --epochs 100
    python train.py --encoder efficientnet-b4 --arch deeplabv3plus --num-classes 5
    python train.py --in-channels 4 --encoder resnet50  # For 4-band imagery (RGB+NIR)

Author: Danesh
License: MIT
"""

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from torchvision import transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import json

# =============================================================================
# SUPPORTED CONFIGURATIONS
# =============================================================================

SUPPORTED_FORMATS = ('.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')

ARCHITECTURES = {
    'unet': smp.Unet,
    'unetplusplus': smp.UnetPlusPlus,
    'manet': smp.MAnet,
    'linknet': smp.Linknet,
    'fpn': smp.FPN,
    'pspnet': smp.PSPNet,
    'pan': smp.PAN,
    'deeplabv3': smp.DeepLabV3,
    'deeplabv3plus': smp.DeepLabV3Plus,
}

ENCODERS = [
    # VGG Family
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
    # ResNet Family
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # ResNeXt Family
    'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d',
    # SE-ResNet Family
    'se_resnet50', 'se_resnet101', 'se_resnet152',
    'se_resnext50_32x4d', 'se_resnext101_32x4d',
    # DenseNet Family
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    # Inception Family
    'inceptionv4', 'inceptionresnetv2',
    # EfficientNet Family
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    # MobileNet Family
    'mobilenet_v2', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_small_100',
    # MiT (Mix Transformer) Family
    'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5',
    # Other
    'dpn68', 'dpn98', 'dpn131',
    'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_008',
    'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_008',
]

LOSSES = {
    'dice': 'DiceLoss',
    'jaccard': 'JaccardLoss',
    'ce': 'CrossEntropyLoss',
    'bce': 'BCEWithLogitsLoss',
    'focal': 'FocalLoss',
    'tversky': 'TverskyLoss',
    'lovasz': 'LovaszLoss',
}

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
}

SCHEDULERS = {
    'plateau': 'ReduceLROnPlateau',
    'step': 'StepLR',
    'cosine': 'CosineAnnealingLR',
    'cosine_warm': 'CosineAnnealingWarmRestarts',
    'exponential': 'ExponentialLR',
    'none': None,
}

# =============================================================================
# DATASET CLASS
# =============================================================================

class SatelliteDataset(Dataset):
    """
    Dataset for satellite/aerial image segmentation.
    
    Supports:
    - Multiple image formats (TIFF, PNG, JPG, etc.)
    - Multi-channel images (RGB, RGB+NIR, multispectral)
    - Binary and multi-class segmentation
    - Patch-based training for large images
    """
    
    def __init__(self, image_dir, mask_dir, patch_size=256, stride=256, 
                 transform=None, num_classes=1, filter_empty=True, in_channels=3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.num_classes = num_classes
        self.filter_empty = filter_empty
        self.in_channels = in_channels
        
        self.images = [f for f in sorted(os.listdir(image_dir)) 
                      if f.lower().endswith(SUPPORTED_FORMATS)]
        
        self.patches = []
        self._prepare_patches()
        
        print(f"\n{'='*50}")
        print(f"Dataset: {image_dir}")
        print(f"{'='*50}")
        print(f"Images found: {len(self.images)}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Stride: {stride}")
        print(f"Input channels: {in_channels}")
        print(f"Number of classes: {num_classes}")
        print(f"Filter empty patches: {filter_empty}")
        print(f"Total patches: {len(self.patches)}")
        print(f"{'='*50}\n")

    def _find_mask_path(self, img_name):
        """Find corresponding mask file."""
        base_name = os.path.splitext(img_name)[0]
        
        mask_path = os.path.join(self.mask_dir, img_name)
        if os.path.exists(mask_path):
            return mask_path
        
        for ext in SUPPORTED_FORMATS:
            mask_path = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(mask_path):
                return mask_path
        
        return os.path.join(self.mask_dir, img_name)

    def _prepare_patches(self):
        for img_name in self.images:
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = self._find_mask_path(img_name)
            
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    w, h = img.size
                    mask_array = np.array(mask) if self.filter_empty else None
                    
                    for y in range(0, max(0, h - self.patch_size + 1), self.stride):
                        for x in range(0, max(0, w - self.patch_size + 1), self.stride):
                            if self.filter_empty:
                                mask_patch = mask_array[y:y+self.patch_size, x:x+self.patch_size]
                                if not np.any(mask_patch > 0):
                                    continue
                            
                            self.patches.append({
                                'image_name': img_name,
                                'x': x,
                                'y': y
                            })
                    
                    # Handle edges
                    if h >= self.patch_size and w >= self.patch_size:
                        if w % self.stride != 0:
                            x = w - self.patch_size
                            for y in range(0, h - self.patch_size + 1, self.stride):
                                if self.filter_empty:
                                    mask_patch = mask_array[y:y+self.patch_size, x:x+self.patch_size]
                                    if not np.any(mask_patch > 0):
                                        continue
                                self.patches.append({'image_name': img_name, 'x': x, 'y': y})
                        
                        if h % self.stride != 0:
                            y = h - self.patch_size
                            for x in range(0, w - self.patch_size + 1, self.stride):
                                if self.filter_empty:
                                    mask_patch = mask_array[y:y+self.patch_size, x:x+self.patch_size]
                                    if not np.any(mask_patch > 0):
                                        continue
                                self.patches.append({'image_name': img_name, 'x': x, 'y': y})
                        
                        if w % self.stride != 0 and h % self.stride != 0:
                            if self.filter_empty:
                                mask_patch = mask_array[h-self.patch_size:h, w-self.patch_size:w]
                                if np.any(mask_patch > 0):
                                    self.patches.append({
                                        'image_name': img_name,
                                        'x': w - self.patch_size,
                                        'y': h - self.patch_size
                                    })
                            else:
                                self.patches.append({
                                    'image_name': img_name,
                                    'x': w - self.patch_size,
                                    'y': h - self.patch_size
                                })
            except Exception as e:
                print(f"Warning: Error processing {img_name}: {e}")
                continue
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        img_name = patch_info['image_name']
        x, y = patch_info['x'], patch_info['y']
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = self._find_mask_path(img_name)
        
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        image_patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
        mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        # Handle different input channels
        if self.in_channels == 3:
            if image_patch.mode != 'RGB':
                image_patch = image_patch.convert('RGB')
        elif self.in_channels == 1:
            if image_patch.mode != 'L':
                image_patch = image_patch.convert('L')
        # For >3 channels, assume image is already in correct format
        
        if self.transform:
            image_patch = self.transform(image_patch)
        
        # Process mask based on number of classes
        mask_array = np.array(mask_patch)
        
        if self.num_classes == 1:
            # Binary segmentation
            mask_tensor = torch.from_numpy(mask_array).float()
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor[:, :, 0]
            mask_tensor = (mask_tensor > 0).float().unsqueeze(0)
        else:
            # Multi-class segmentation (mask contains class indices 0, 1, 2, ...)
            if mask_array.ndim == 3:
                mask_array = mask_array[:, :, 0]
            mask_tensor = torch.from_numpy(mask_array).long()
        
        return image_patch, mask_tensor, (img_name, x, y)

# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    def __init__(self, model, criterion, optimizer, device, scaler, use_amp=True, num_classes=1):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.use_amp = use_amp
        self.num_classes = num_classes
    
    def train_step(self, images, masks):
        images = images.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)
        
        if self.use_amp:
            with autocast(device_type='cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item(), outputs
    
    @torch.no_grad()
    def validate_step(self, images, masks):
        images = images.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)
        
        if self.use_amp:
            with autocast(device_type='cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
        else:
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
        
        return loss.item(), outputs

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_metrics(pred, target, num_classes=1, threshold=0.5):
    """Calculate IoU and Dice score for binary or multi-class segmentation."""
    
    if num_classes == 1:
        # Binary segmentation
        pred = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        dice = (2 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
    else:
        # Multi-class segmentation
        pred = torch.argmax(pred, dim=1)
        ious = []
        dices = []
        
        for cls in range(num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            
            iou = (intersection + 1e-7) / (union + 1e-7)
            dice = (2 * intersection + 1e-7) / (pred_cls.sum() + target_cls.sum() + 1e-7)
            
            ious.append(iou.item())
            dices.append(dice.item())
        
        iou = np.mean(ious)
        dice = np.mean(dices)
    
    return iou if isinstance(iou, float) else iou.item(), dice if isinstance(dice, float) else dice.item()

def reconstruct_predictions(predictions, patch_info, original_sizes, patch_size, num_classes=1):
    """Reconstruct full images from patches."""
    reconstructed = {}
    counts = {}
    
    predictions = torch.stack(predictions) if isinstance(predictions, list) else predictions
    
    for idx, (img_name, x, y) in enumerate(patch_info):
        if img_name not in reconstructed:
            h, w = original_sizes[img_name]
            if num_classes == 1:
                reconstructed[img_name] = torch.zeros((1, h, w), device='cpu')
            else:
                reconstructed[img_name] = torch.zeros((num_classes, h, w), device='cpu')
            counts[img_name] = torch.zeros((1, h, w), device='cpu')
        
        pred = predictions[idx]
        
        if num_classes == 1:
            pred = pred.squeeze(0)
            reconstructed[img_name][:, y:y+patch_size, x:x+patch_size] += pred
        else:
            reconstructed[img_name][:, y:y+patch_size, x:x+patch_size] += pred
        
        counts[img_name][:, y:y+patch_size, x:x+patch_size] += 1
    
    for img_name in reconstructed:
        reconstructed[img_name] = reconstructed[img_name] / counts[img_name].clamp(min=1)
    
    return reconstructed

def get_loss_function(loss_name, num_classes, **kwargs):
    """Get loss function based on name and number of classes."""
    mode = 'binary' if num_classes == 1 else 'multiclass'
    
    if loss_name == 'dice':
        return smp.losses.DiceLoss(mode=mode)
    elif loss_name == 'jaccard':
        return smp.losses.JaccardLoss(mode=mode)
    elif loss_name == 'focal':
        return smp.losses.FocalLoss(mode=mode, alpha=kwargs.get('focal_alpha', 0.25), 
                                     gamma=kwargs.get('focal_gamma', 2.0))
    elif loss_name == 'tversky':
        return smp.losses.TverskyLoss(mode=mode, alpha=kwargs.get('tversky_alpha', 0.5),
                                       beta=kwargs.get('tversky_beta', 0.5))
    elif loss_name == 'lovasz':
        return smp.losses.LovaszLoss(mode=mode)
    elif loss_name == 'ce':
        return torch.nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    else:
        return smp.losses.DiceLoss(mode=mode)

def get_scheduler(optimizer, scheduler_name, epochs, **kwargs):
    """Get learning rate scheduler."""
    if scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=kwargs.get('factor', 0.5), 
            patience=kwargs.get('patience', 5), verbose=True
        )
    elif scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get('step_size', 10), 
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'cosine_warm':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=kwargs.get('t0', 10), T_mult=kwargs.get('t_mult', 2)
        )
    elif scheduler_name == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.95))
    else:
        return None

def print_config(args):
    """Print configuration summary."""
    print("\n" + "="*60)
    print(" SATELLITE IMAGE SEGMENTATION - CONFIGURATION")
    print("="*60)
    print(f"\n📁 Data Paths:")
    print(f"   Train:      {args.train_dir} / {args.train_mask_dir}")
    print(f"   Validation: {args.val_dir} / {args.val_mask_dir}")
    print(f"   Test:       {args.test_dir} / {args.test_mask_dir}")
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Architecture: {args.arch.upper()}")
    print(f"   Encoder:      {args.encoder}")
    print(f"   Pretrained:   {args.pretrained}")
    print(f"   In Channels:  {args.in_channels}")
    print(f"   Classes:      {args.num_classes}")
    
    print(f"\n⚙️  Training Hyperparameters:")
    print(f"   Epochs:       {args.epochs}")
    print(f"   Batch Size:   {args.batch_size}")
    print(f"   Learning Rate:{args.lr}")
    print(f"   Weight Decay: {args.weight_decay}")
    print(f"   Optimizer:    {args.optimizer}")
    print(f"   Scheduler:    {args.scheduler}")
    print(f"   Loss:         {args.loss}")
    
    print(f"\n📐 Patch Settings:")
    print(f"   Patch Size:   {args.patch_size}")
    print(f"   Stride:       {args.stride}")
    print(f"   Filter Empty: {args.filter_empty}")
    
    print(f"\n💻 System:")
    print(f"   Device:       {args.device}")
    print(f"   Workers:      {args.num_workers}")
    print(f"   Mixed Prec.:  {args.amp}")
    
    print("="*60 + "\n")

# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Satellite Image Segmentation Training Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    data_group = parser.add_argument_group('Data Paths')
    data_group.add_argument('--train-dir', type=str, default='data/train',
                           help='Training images directory')
    data_group.add_argument('--train-mask-dir', type=str, default='data/train_masks',
                           help='Training masks directory')
    data_group.add_argument('--val-dir', type=str, default='data/val',
                           help='Validation images directory')
    data_group.add_argument('--val-mask-dir', type=str, default='data/val_masks',
                           help='Validation masks directory')
    data_group.add_argument('--test-dir', type=str, default='data/test',
                           help='Test images directory')
    data_group.add_argument('--test-mask-dir', type=str, default='data/test_masks',
                           help='Test masks directory')
    data_group.add_argument('--output-dir', type=str, default='outputs',
                           help='Output directory for predictions and checkpoints')
    
    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--arch', type=str, default='unet',
                            choices=list(ARCHITECTURES.keys()),
                            help='Segmentation architecture')
    model_group.add_argument('--encoder', type=str, default='resnet50',
                            choices=ENCODERS,
                            help='Encoder backbone')
    model_group.add_argument('--pretrained', type=str, default='imagenet',
                            choices=['imagenet', 'ssl', 'swsl', 'none'],
                            help='Pretrained weights (use "none" for random init)')
    model_group.add_argument('--in-channels', type=int, default=3,
                            help='Number of input channels (3=RGB, 4=RGB+NIR, etc.)')
    model_group.add_argument('--num-classes', type=int, default=1,
                            help='Number of output classes (1=binary, >1=multi-class)')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, default=8,
                            help='Batch size')
    train_group.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=1e-4,
                            help='Weight decay (L2 regularization)')
    train_group.add_argument('--momentum', type=float, default=0.9,
                            help='Momentum for SGD optimizer')
    
    # Optimizer
    optim_group = parser.add_argument_group('Optimizer')
    optim_group.add_argument('--optimizer', type=str, default='adamw',
                            choices=list(OPTIMIZERS.keys()),
                            help='Optimizer type')
    
    # Scheduler
    sched_group = parser.add_argument_group('Learning Rate Scheduler')
    sched_group.add_argument('--scheduler', type=str, default='plateau',
                            choices=list(SCHEDULERS.keys()),
                            help='Learning rate scheduler')
    sched_group.add_argument('--scheduler-patience', type=int, default=5,
                            help='Patience for ReduceLROnPlateau')
    sched_group.add_argument('--scheduler-factor', type=float, default=0.5,
                            help='Factor for ReduceLROnPlateau')
    sched_group.add_argument('--scheduler-step-size', type=int, default=10,
                            help='Step size for StepLR')
    sched_group.add_argument('--scheduler-gamma', type=float, default=0.1,
                            help='Gamma for StepLR/ExponentialLR')
    sched_group.add_argument('--scheduler-t0', type=int, default=10,
                            help='T_0 for CosineAnnealingWarmRestarts')
    sched_group.add_argument('--scheduler-t-mult', type=int, default=2,
                            help='T_mult for CosineAnnealingWarmRestarts')
    
    # Loss function
    loss_group = parser.add_argument_group('Loss Function')
    loss_group.add_argument('--loss', type=str, default='dice',
                           choices=list(LOSSES.keys()),
                           help='Loss function')
    loss_group.add_argument('--focal-alpha', type=float, default=0.25,
                           help='Alpha for Focal Loss')
    loss_group.add_argument('--focal-gamma', type=float, default=2.0,
                           help='Gamma for Focal Loss')
    loss_group.add_argument('--tversky-alpha', type=float, default=0.5,
                           help='Alpha for Tversky Loss')
    loss_group.add_argument('--tversky-beta', type=float, default=0.5,
                           help='Beta for Tversky Loss')
    
    # Patch settings
    patch_group = parser.add_argument_group('Patch Settings')
    patch_group.add_argument('--patch-size', type=int, default=256,
                            help='Patch size for training')
    patch_group.add_argument('--stride', type=int, default=128,
                            help='Stride for patch extraction')
    patch_group.add_argument('--filter-empty', action='store_true', default=True,
                            help='Filter out patches without annotations')
    patch_group.add_argument('--no-filter-empty', action='store_false', dest='filter_empty',
                            help='Include all patches')
    
    # System settings
    sys_group = parser.add_argument_group('System Settings')
    sys_group.add_argument('--device', type=str, default='auto',
                          choices=['auto', 'cuda', 'cpu', 'mps'],
                          help='Device to use')
    sys_group.add_argument('--num-workers', type=int, default=4,
                          help='Number of data loading workers')
    sys_group.add_argument('--amp', action='store_true', default=True,
                          help='Use automatic mixed precision')
    sys_group.add_argument('--no-amp', action='store_false', dest='amp',
                          help='Disable automatic mixed precision')
    sys_group.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    
    # Checkpointing
    ckpt_group = parser.add_argument_group('Checkpointing')
    ckpt_group.add_argument('--resume', type=str, default=None,
                           help='Path to checkpoint to resume from')
    ckpt_group.add_argument('--save-every', type=int, default=10,
                           help='Save checkpoint every N epochs')
    
    # Inference
    infer_group = parser.add_argument_group('Inference')
    infer_group.add_argument('--threshold', type=float, default=0.5,
                            help='Threshold for binary prediction')
    infer_group.add_argument('--test-only', action='store_true',
                            help='Only run inference on test set')
    infer_group.add_argument('--weights', type=str, default=None,
                            help='Path to model weights for inference')
    
    return parser.parse_args()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    args.device = device
    
    # Enable cuDNN benchmark
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Print configuration
    print_config(args)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.arch}_{args.encoder}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        config_dict = vars(args).copy()
        config_dict['device'] = str(config_dict['device'])
        json.dump(config_dict, f, indent=2)
    
    # Create model
    print(f"Creating model: {args.arch.upper()} with {args.encoder} encoder...")
    encoder_weights = args.pretrained if args.pretrained != 'none' else None
    
    model = ARCHITECTURES[args.arch](
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        in_channels=args.in_channels,
        classes=args.num_classes,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = get_loss_function(
        args.loss, args.num_classes,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta
    )
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = OPTIMIZERS['sgd'](
            model.parameters(), lr=args.lr, 
            momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        optimizer = OPTIMIZERS[args.optimizer](
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer, args.scheduler, args.epochs,
        factor=args.scheduler_factor, patience=args.scheduler_patience,
        step_size=args.scheduler_step_size, gamma=args.scheduler_gamma,
        t0=args.scheduler_t0, t_mult=args.scheduler_t_mult
    )
    
    # Create scaler for AMP
    scaler = GradScaler() if args.amp and device.type == 'cuda' else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    # Test only mode
    if args.test_only:
        if args.weights:
            checkpoint = torch.load(args.weights, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        run_inference(model, args, device, output_dir)
        return
    
    # Create transforms
    if args.in_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # For non-RGB images, just convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SatelliteDataset(
        args.train_dir, args.train_mask_dir,
        patch_size=args.patch_size, stride=args.stride,
        transform=transform, num_classes=args.num_classes,
        filter_empty=args.filter_empty, in_channels=args.in_channels
    )
    
    val_dataset = SatelliteDataset(
        args.val_dir, args.val_mask_dir,
        patch_size=args.patch_size, stride=args.stride,
        transform=transform, num_classes=args.num_classes,
        filter_empty=args.filter_empty, in_channels=args.in_channels
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
    )
    
    # Create trainer
    trainer = Trainer(model, criterion, optimizer, device, scaler, 
                     use_amp=args.amp, num_classes=args.num_classes)
    
    # Training loop
    print("\n" + "="*60)
    print(" TRAINING STARTED")
    print("="*60 + "\n")
    
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': [], 'lr': []}
    
    for epoch in range(start_epoch, args.epochs):
        # Training
        model.train()
        train_losses = []
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]') as pbar:
            for images, masks, _ in pbar:
                loss, _ = trainer.train_step(images, masks)
                train_losses.append(loss)
                pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_ious = []
        val_dices = []
        
        with tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Valid]') as pbar:
            for images, masks, _ in pbar:
                loss, outputs = trainer.validate_step(images, masks)
                val_losses.append(loss)
                
                iou, dice = calculate_metrics(outputs.cpu(), masks, args.num_classes)
                val_ious.append(iou)
                val_dices.append(dice)
                
                pbar.set_postfix({'loss': f'{loss:.4f}', 'iou': f'{iou:.4f}'})
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_iou = sum(val_ious) / len(val_ious)
        avg_val_dice = sum(val_dices) / len(val_dices)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        history['val_dice'].append(avg_val_dice)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss:   {avg_val_loss:.4f}')
        print(f'  Val IoU:    {avg_val_iou:.4f}')
        print(f'  Val Dice:   {avg_val_dice:.4f}')
        print(f'  LR:         {current_lr:.2e}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_iou': avg_val_iou,
                'config': vars(args),
            }, os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
            print(f'  ✓ New best model saved!')
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'))
        
        print('-' * 40)
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETED")
    print("="*60)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Run inference on test set
    print("\nRunning inference on test set...")
    checkpoint = torch.load(os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    run_inference(model, args, device, output_dir)


def run_inference(model, args, device, output_dir):
    """Run inference on test set."""
    model.eval()
    
    if args.in_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = SatelliteDataset(
        args.test_dir, args.test_mask_dir,
        patch_size=args.patch_size, stride=args.stride,
        transform=transform, num_classes=args.num_classes,
        filter_empty=False, in_channels=args.in_channels
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    predictions = []
    patch_info_list = []
    
    with torch.no_grad():
        for images, _, patch_info in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            
            if args.amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            if args.num_classes == 1:
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
            
            outputs = outputs.cpu()
            predictions.extend(outputs)
            patch_info_list.extend(list(zip(*patch_info)))
    
    # Get original image sizes
    original_sizes = {}
    for img_name in os.listdir(args.test_dir):
        if img_name.lower().endswith(SUPPORTED_FORMATS):
            with Image.open(os.path.join(args.test_dir, img_name)) as img:
                original_sizes[img_name] = img.size[::-1]
    
    # Reconstruct and save predictions
    reconstructed_preds = reconstruct_predictions(
        predictions, patch_info_list, original_sizes, args.patch_size, args.num_classes
    )
    
    pred_dir = os.path.join(output_dir, 'predictions')
    for img_name, pred in reconstructed_preds.items():
        if args.num_classes == 1:
            # Binary: threshold and save
            pred_image = (pred > args.threshold).float() * 255
            pred_image = pred_image.squeeze().numpy().astype(np.uint8)
        else:
            # Multi-class: save class indices
            pred_image = torch.argmax(pred, dim=0).numpy().astype(np.uint8)
        
        output_name = os.path.splitext(img_name)[0] + '.png'
        Image.fromarray(pred_image).save(os.path.join(pred_dir, output_name))
    
    print(f"\nPredictions saved to: {pred_dir}")


if __name__ == '__main__':
    main()
