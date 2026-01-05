# Weights Directory

Store pretrained encoder weights here (optional).

## Automatic Download

Pretrained weights are downloaded automatically when using `--pretrained imagenet`.

## Manual Download (for offline use)

```bash
# ResNet50
wget https://download.pytorch.org/models/resnet50-19c8e357.pth

# EfficientNet-B4
# Downloaded via timm library automatically
```

## Trained Models

After training, your model checkpoints will be saved in `outputs/<run_name>/checkpoints/`.
