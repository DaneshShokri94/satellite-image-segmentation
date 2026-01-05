# Data Directory

Place your satellite/aerial imagery here.

## Directory Structure

```
data/
├── train/              # Training images
│   ├── image_001.tif
│   ├── image_002.png
│   └── ...
├── train_masks/        # Training masks
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── val/                # Validation images
│   └── ...
├── val_masks/          # Validation masks
│   └── ...
├── test/               # Test images
│   └── ...
└── test_masks/         # Test masks
    └── ...
```

## Supported Formats

- TIFF (`.tiff`, `.tif`) - Recommended for satellite data
- PNG (`.png`) - Recommended for masks
- JPEG (`.jpg`, `.jpeg`)
- BMP (`.bmp`)
- WebP (`.webp`)

## Mask Format

### Binary Segmentation (1 class)
- Background: 0 (black)
- Target: 255 (white) or 1

### Multi-class Segmentation
- Class 0: pixel value 0
- Class 1: pixel value 1
- Class 2: pixel value 2
- ... and so on

## Example Use Cases

### Water Detection
- Images: RGB satellite imagery
- Masks: Binary (0=land, 255=water)

### Land Cover Classification
- Images: RGB or multispectral
- Masks: Multi-class (0=background, 1=water, 2=forest, 3=urban, 4=agriculture)

### Ice Classification (SWOT-like)
- Images: SAR or optical imagery
- Masks: Binary (0=water, 255=ice)

## Tips

1. Images and masks must have the same base filename
2. Large images are automatically split into patches
3. Empty patches (no labels) are filtered by default
4. Use PNG for masks (lossless compression)
