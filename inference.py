#!/usr/bin/env python3
"""
Satellite Image Segmentation - Inference Script
================================================
Run inference on new satellite/aerial images.

Usage:
    python inference.py --weights best_model.pth --input images/ --output predictions/
    python inference.py --weights best_model.pth --input single_image.tif

Author: Danesh
License: MIT
"""

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import argparse
import numpy as np
from torchvision import transforms
from torch.amp import autocast
from tqdm import tqdm
import json

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


class InferenceDataset(Dataset):
    """Dataset for inference on new images."""
    
    def __init__(self, image_paths, patch_size=256, stride=128, transform=None, in_channels=3):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.in_channels = in_channels
        
        self.patches = []
        self._prepare_patches()
    
    def _prepare_patches(self):
        for img_path in self.image_paths:
            with Image.open(img_path) as img:
                w, h = img.size
                img_name = os.path.basename(img_path)
                
                for y in range(0, max(0, h - self.patch_size + 1), self.stride):
                    for x in range(0, max(0, w - self.patch_size + 1), self.stride):
                        self.patches.append({
                            'image_path': img_path,
                            'image_name': img_name,
                            'x': x,
                            'y': y,
                            'size': (w, h)
                        })
                
                if h >= self.patch_size and w >= self.patch_size:
                    if w % self.stride != 0:
                        x = w - self.patch_size
                        for y in range(0, h - self.patch_size + 1, self.stride):
                            self.patches.append({
                                'image_path': img_path,
                                'image_name': img_name,
                                'x': x, 'y': y,
                                'size': (w, h)
                            })
                    
                    if h % self.stride != 0:
                        y = h - self.patch_size
                        for x in range(0, w - self.patch_size + 1, self.stride):
                            self.patches.append({
                                'image_path': img_path,
                                'image_name': img_name,
                                'x': x, 'y': y,
                                'size': (w, h)
                            })
                    
                    if w % self.stride != 0 and h % self.stride != 0:
                        self.patches.append({
                            'image_path': img_path,
                            'image_name': img_name,
                            'x': w - self.patch_size,
                            'y': h - self.patch_size,
                            'size': (w, h)
                        })
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        
        image = Image.open(patch_info['image_path'])
        x, y = patch_info['x'], patch_info['y']
        
        image_patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        if self.in_channels == 3:
            if image_patch.mode != 'RGB':
                image_patch = image_patch.convert('RGB')
        elif self.in_channels == 1:
            if image_patch.mode != 'L':
                image_patch = image_patch.convert('L')
        
        if self.transform:
            image_patch = self.transform(image_patch)
        
        return image_patch, (patch_info['image_name'], x, y)


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


def load_model(weights_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    arch = config.get('arch', 'unet')
    encoder = config.get('encoder', 'resnet50')
    in_channels = config.get('in_channels', 3)
    num_classes = config.get('num_classes', 1)
    
    print(f"Loading model: {arch.upper()} with {encoder} encoder")
    print(f"Input channels: {in_channels}, Classes: {num_classes}")
    
    model = ARCHITECTURES[arch](
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Satellite Image Segmentation - Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights (.pth file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Output directory')
    parser.add_argument('--patch-size', type=int, default=None,
                       help='Patch size (default: from model config)')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride (default: from model config)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary prediction')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Data loading workers')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use mixed precision')
    parser.add_argument('--save-prob', action='store_true',
                       help='Save probability maps')
    parser.add_argument('--save-color', action='store_true',
                       help='Save colored output (multi-class)')
    
    return parser.parse_args()


# Default color palette for multi-class visualization
DEFAULT_COLORS = [
    (0, 0, 0),       # 0: Background - Black
    (0, 0, 255),     # 1: Blue
    (0, 255, 0),     # 2: Green
    (255, 0, 0),     # 3: Red
    (255, 255, 0),   # 4: Yellow
    (255, 0, 255),   # 5: Magenta
    (0, 255, 255),   # 6: Cyan
    (128, 0, 0),     # 7: Dark Red
    (0, 128, 0),     # 8: Dark Green
    (0, 0, 128),     # 9: Dark Blue
]


def apply_color_map(mask, colors=None):
    """Apply color map to multi-class mask."""
    if colors is None:
        colors = DEFAULT_COLORS
    
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(colors):
        colored[mask == class_idx] = color
    
    return colored


def main():
    args = parse_args()
    
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
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.weights, device)
    
    # Get settings from config
    patch_size = args.patch_size or config.get('patch_size', 256)
    stride = args.stride or config.get('stride', 128)
    in_channels = config.get('in_channels', 3)
    num_classes = config.get('num_classes', 1)
    
    print(f"Patch size: {patch_size}, Stride: {stride}")
    
    # Get input images
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = [
            os.path.join(args.input, f) 
            for f in sorted(os.listdir(args.input))
            if f.lower().endswith(SUPPORTED_FORMATS)
        ]
    
    print(f"Found {len(image_paths)} images")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create transform
    if in_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    # Create dataset and dataloader
    dataset = InferenceDataset(image_paths, patch_size, stride, transform, in_channels)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Total patches: {len(dataset)}")
    
    # Run inference
    predictions = []
    patch_info_list = []
    
    with torch.no_grad():
        for images, patch_info in tqdm(dataloader, desc='Processing'):
            images = images.to(device)
            
            if args.amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            if num_classes == 1:
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
            
            outputs = outputs.cpu()
            predictions.extend(outputs)
            patch_info_list.extend(list(zip(*patch_info)))
    
    # Get original sizes
    original_sizes = {}
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        with Image.open(img_path) as img:
            original_sizes[img_name] = img.size[::-1]
    
    # Reconstruct predictions
    print("Reconstructing full images...")
    reconstructed = reconstruct_predictions(
        predictions, patch_info_list, original_sizes, patch_size, num_classes
    )
    
    # Save predictions
    print("Saving predictions...")
    for img_name, pred in reconstructed.items():
        base_name = os.path.splitext(img_name)[0]
        
        if num_classes == 1:
            # Binary segmentation
            if args.save_prob:
                pred_image = (pred.squeeze().numpy() * 255).astype(np.uint8)
                Image.fromarray(pred_image).save(os.path.join(args.output, f'{base_name}_prob.png'))
            
            pred_binary = (pred > args.threshold).float() * 255
            pred_binary = pred_binary.squeeze().numpy().astype(np.uint8)
            Image.fromarray(pred_binary).save(os.path.join(args.output, f'{base_name}.png'))
        else:
            # Multi-class segmentation
            pred_classes = torch.argmax(pred, dim=0).numpy().astype(np.uint8)
            Image.fromarray(pred_classes).save(os.path.join(args.output, f'{base_name}.png'))
            
            if args.save_color:
                colored = apply_color_map(pred_classes)
                Image.fromarray(colored).save(os.path.join(args.output, f'{base_name}_color.png'))
            
            if args.save_prob:
                # Save confidence map
                confidence = torch.max(pred, dim=0)[0].numpy()
                confidence = (confidence * 255).astype(np.uint8)
                Image.fromarray(confidence).save(os.path.join(args.output, f'{base_name}_confidence.png'))
    
    print(f"\nPredictions saved to: {args.output}")
    print(f"Total images processed: {len(reconstructed)}")


if __name__ == '__main__':
    main()
