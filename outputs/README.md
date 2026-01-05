# Outputs Directory

Training outputs will be saved here automatically.

## Structure

After training, each run creates a timestamped folder:

```
outputs/
└── unet_resnet50_20240115_143022/
    ├── config.json           # Training configuration
    ├── history.json          # Loss and metrics per epoch
    ├── checkpoints/
    │   ├── best_model.pth    # Best validation loss
    │   └── checkpoint_epoch_50.pth
    └── predictions/
        ├── image_001.png
        └── ...
```

## Loading a Trained Model

```python
import torch

checkpoint = torch.load('outputs/run_name/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Inference with Trained Model

```bash
python inference.py --weights outputs/run_name/checkpoints/best_model.pth --input new_images/
```
