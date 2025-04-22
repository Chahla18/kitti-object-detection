# YOLOv8 Object Detection on KITTI Dataset

This repository contains an implementation of YOLOv8 for object detection on the KITTI autonomous driving dataset. The project focuses on optimizing detection performance across different object classes with varying frequencies and occlusion levels.

## Dataset

The [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php) consists of 7,481 training images and 7,518 test images, comprising a total of 80,256 labeled objects. The dataset contains annotations for cars, pedestrians, cyclists, and other road objects captured from a moving vehicle.

Key dataset characteristics:
- Significant class imbalance (Cars: 55.42%, Pedestrians: 8.65%, Cyclists: 3.14%)
- Various occlusion levels and object sizes
- Objects captured at different orientations and distances

## Dataset Preprocessing

The preprocessing pipeline includes:
- Converting KITTI format to YOLO format
- Creating a class-balanced train/validation split
- Generating bounding box statistics and visualizations
- Handling occlusion levels and truncation information

## Model Architecture

The project implements YOLOv8, a state-of-the-art object detection architecture from Ultralytics. Two model variants were tested:
- YOLOv8-M: Medium-sized model with a good speed/accuracy trade-off
- Faster R-CNN

## Features

- **Class Weighting**: Implements custom class weights to address dataset imbalance
- **Augmentation Strategy**: Uses domain-specific augmentations suitable for autonomous driving scenarios
- **Occlusion Handling**: Optimized detection of partially visible objects
- **Experiment Tracking**: Full integration with Weights & Biases for experiment tracking
- **Model Versioning**: Automatic checkpoint saving to Hugging Face Hub

## Results

Performance metrics for our best model until now:

| Class          | mAP50 | mAP50-95 | Precision | Recall |
|----------------|-------|----------|-----------|--------|
| Car            | 0.916 | 0.397    | 0.886     | 0.854  |
| Pedestrian     | 0.666 | 0.222    | 0.830     | 0.594  |
| Cyclist        | 0.643 | 0.230    | 0.688     | 0.639  |
| Truck          | 0.950 | 0.417    | 0.876     | 0.911  |
| Van            | 0.803 | 0.344    | 0.782     | 0.729  |
| Tram           | 0.852 | 0.318    | 0.744     | 0.816  |
| Person_sitting | 0.533 | 0.203    | 0.625     | 0.347  |
| Misc           | 0.646 | 0.266    | 0.779     | 0.577  |
| **Overall**    | **0.683** | **0.270** | **0.743** | **0.613** |

## Key Insights

Our experimentation led to several important findings:
1. Removing the "DontCare" class improved overall metrics
2. Class weights significantly enhanced performance on minority classes
3. Reducing rotation augmentation to realistic angles (5° vs 45°) improved results
4. Extended training (100 epochs) was necessary for optimal convergence

## How to Run

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)

### Installation
```bash
git clone https://github.com/Chahla18/kitti-object-detection.git
cd kitti-object-detection
pip install -r requirements.txt
```

### Dataset Download and Model Training
```bash
python yolov8_advanced.py
```

## Project Structure
```
├── data/                   # Data handling utilities
├── data_exploration.py     # Creating visuals to explore the data
├── yolov8_advanced.py      # Training and Evaluation script
└── requirements.txt        # Dependencies
```

## Visualizations

The repository includes visualizations for:
- Bounding box size distributions per class
- Object orientation distributions
- Occlusion level statistics
- Class distribution analysis

## Acknowledgments

- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/) - for providing the dataset
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - for the detection framework
- [Weights & Biases](https://wandb.ai/) - for experiment tracking
- [Hugging Face](https://huggingface.co/) - for model hosting