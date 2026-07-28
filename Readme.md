# YOLOv3 in PyTorch

A clean and minimal implementation of YOLOv3 (You Only Look Once version 3) in PyTorch for object detection. This implementation focuses on clarity and readability while maintaining the core YOLOv3 architecture.

## Features

- Implementation of YOLOv3 architecture from scratch in PyTorch
- Support for Pascal VOC dataset (easily adaptable to COCO)
- Multi-scale training and prediction
- Custom loss function implementation (YOLO loss)
- Non-Maximum Suppression (NMS) for post-processing
- Mean Average Precision (mAP) calculation for evaluation
- Training and validation scripts
- Pretrained weights support (Pascal VOC)
- Visualization utilities for debugging and visualization

## Project Structure

```
YOLO-v3_From_Scratch/
│
├── model.py          # YOLOv3 architecture definition
├── loss.py           # YOLO loss function implementation
├── dataset.py        # Dataset loading and preprocessing
├── utils.py          # Utility functions (NLP, mAP, plotting, etc.)
├── train.py          # Main training script
├── config.py         # Configuration file (hyperparameters, paths, etc.)
├── Readme.md         # This file
└── requirements.txt  # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ifaz2611/YOLO-v3_From_Scratch
cd YOLO-v3_From_Scratch
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- torch
- matplotlib
- pandas
- tqdm
- albumentations

## Dataset Preparation

This implementation uses the Pascal VOC dataset format.

1. Download the Pascal VOC dataset from [Kaggle](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video)

2. Extract the dataset into the main project directory. The expected structure is:
```
YOLO-v3_From_Scratch/
├── pascal_voc/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── train.csv
├── test.csv
└── ... (source files)
```

3. The CSV files (`train.csv` and `test.csv`) should contain annotations in the format:
```
image_path, x_center, y_center, width, height, class_label
```
Where coordinates are normalized to [0, 1] relative to image dimensions.

## Configuration

Edit `config.py` to adjust training parameters:

```python
# Dataset configuration
DATASET = "pascal_voc"  # or "coco"
NUM_CLASSES = 20        # Number of classes in your dataset
IMAGE_SIZE = 416        # Input image size

# Training hyperparameters
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 100
WEIGHT_DECAY = 5e-4

# Model thresholds
CONF_THRESHOLD = 0.4        # Object confidence threshold
NMS_IOU_THRESH = 0.45       # NMS IoU threshold
MAP_IOU_THRESH = 0.5        # mAP IoU threshold

# Anchor boxes (rescaled to [0, 1] based on 416x416 image size)
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],  # 13x13 scale
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],  # 26x26 scale
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],  # 52x52 scale
]

# Other settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False    # Set to True to load checkpoint
SAVE_MODEL = False    # Set to True to save checkpoint
CHECKPOINT_FILE = "checkpoint.pth.tar"
```

## Training

To start training:
```bash
python train.py
```

Training features:
- Mixed precision training (using torch.cuda.amp)
- Progress bar with tqdm
- Automatic checkpoint saving (when SAVE_MODEL=True)
- Learning rate and weight decay as specified in config
- Batch processing with configurable number of workers

## Evaluation

The training script includes validation during training. After training, you can evaluate using:

```python
# Example usage (can be added to a separate eval script)
import torch
from model import YOLOv3
from utils import get_evaluation_bboxes, mean_average_precision
from utils import load_checkpoint
import config

model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

# Get predictions and ground truth
pred_boxes, true_boxes = get_evaluation_bboxes(
    test_loader, model, 
    iou_threshold=config.NMS_IOU_THRESH,
    threshold=config.CONF_THRESHOLD,
    box_format="midpoint",
    device=config.DEVICE,
)

# Calculate mAP
mean_avg_prec = mean_average_precision(
    pred_boxes, true_boxes,
    iou_threshold=config.MAP_IOU_THRESH,
    box_format="midpoint",
    num_classes=config.NUM_CLASSES,
)
print(f"mAP: {mean_avg_prec}")
```

## Results

After training on Pascal VOC dataset:
- YOLOv3 (Pascal VOC): ~78.2% mAP @ 0.5 IoU
- Confidence threshold: 0.2
- NMS IoU threshold: 0.45

## Model Architecture

This implementation follows the original YOLOv3 architecture:

1. **Darknet-53 Backbone**: Feature extractor consisting of convolutional layers and residual blocks
2. **Multi-scale Predictions**: Three detection scales at different resolutions:
   - Large objects: 13x13 grid
   - Medium objects: 26x26 grid  
   - Small objects: 52x52 grid
3. **Anchor Boxes**: Prior boxes for each scale to handle different aspect ratios
4. **Feature Pyramid Network**: Upsampling and concatenation for better feature reuse

Key components in `model.py`:
- `CNNBlock`: Basic convolutional block with batch norm and LeakyReLU
- `ResidualBlock`: Residual blocks used in Darknet-53
- `ScalePrediction`: Prediction head for each scale
- `YOLOv3`: Main model architecture

## Loss Function

The YOLO loss (`loss.py`) combines three components:
1. **Objectness loss**: Binary cross-entropy for object presence
2. **Classification loss**: Cross-entropy for class prediction
3. **Localization loss**: Mean squared error for bounding box coordinates

## Key Implementation Details

### Configuration (`config.py`)
- Centralized configuration for easy modification
- Dataset paths, hyperparameters, model settings, and transforms
- Augmentation pipeline using Albumentations library

### Dataset Handling (`dataset.py`)
- Custom VOCDataset class for loading images and labels
- Data augmentation (random crops, flips, color adjustments, etc.)
- Label encoding to YOLO format (cell coordinates, offsets, class labels)
- Ignore mechanism for boxes that match anchors too well but aren't the best match

### Utilities (`utils.py`)
- Non-Maximum Suppression (NMS)
- Mean Average Precision (mAP) calculation
- Box conversion utilities (cells to bounding boxes)
- Model checkpointing (save/load)
- Training utilities (accuracy checking, plotting examples)
- Data loading functions

## Training Process

1. **Data Loading**: Images and labels are loaded and augmented
2. **Forward Pass**: Images pass through YOLOv3 to get predictions at three scales
3. **Loss Calculation**: YOLO loss computed for each scale
4. **Backpropagation**: Gradients computed and optimizer updates weights
5. **Evaluation**: Periodic mAP calculation on validation set
6. **Checkpointing**: Model saved periodically (if enabled)

## Customization

### Changing Dataset
To use a different dataset:
1. Modify `config.py`: change `DATASET`, `NUM_CLASSES`, and paths
2. Prepare dataset in YOLO format (normalized coordinates)
3. Update anchor boxes if needed (should match your dataset's object scales)

### Changing Model Size
The architecture is defined in `model.py`'s `config` list. To modify:
- Adjust the number of filters, layers, or repeats
- Change the scale prediction layers (`"S"` markers)
- Modify upsampling (`"U"` markers) as needed

## References

- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) by Joseph Redmon, Ali Farhadi
- Original YOLOv3 implementation: https://pjreddie.com/darknet/yolo/
- PyTorch YOLOv3 implementations that inspired this project:
  - https://github.com/eriklindernoren/PyTorch-YOLOv3
  - https://github.com/eriklindernoren/PyTorch-YOLOv3

## License

This project is for educational purposes. Please refer to the original YOLOv3 license for any commercial use.

## Acknowledgments

- Inspired by various PyTorch YOLO implementations
- Uses Albumentations for efficient image augmentation
- Built with PyTorch deep learning framework