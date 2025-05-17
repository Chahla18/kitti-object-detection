# YOLOv8 Object Detection on KITTI Dataset

This repository contains an implementation of YOLOv8 for object detection on the KITTI autonomous driving dataset. The project focuses on optimizing detection performance across different object classes with varying frequencies and occlusion levels.

## 📋 Project Overview

We evaluated two distinct approaches for object detection in driving scenes:
- **YOLOv8** (single-stage detector)
- **Faster R-CNN** (two-stage detector)

Our study demonstrated that YOLOv8 offers superior performance for object detection in autonomous driving contexts, both in terms of accuracy and inference speed.

## 🔍 Dataset Analysis

The [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php) consists of 7,481 training images of road scenes captured from a moving vehicle. The dataset contains annotations for cars, pedestrians, cyclists, and other road objects captured from a moving vehicle. Our analysis revealed:

- Significant class imbalance (Cars: 55.4%, Pedestrians: 8.7%, Cyclists: 3.1%)
- Bimodal distribution of vehicle orientations corresponding to main road directions
- Various occlusion levels and truncations at image borders

## 🛠️ Methodology

### Data Preprocessing
- Stratified sampling (80/20 split) to ensure balanced representation of minority classes
- Exclusion of the "DontCare" class to improve learning quality
- Format conversion specific to each architecture (normalized coordinates for YOLOv8, absolute coordinates for Faster R-CNN)

### Model Architectures
- **YOLOv8m**: 25.9M parameters, 640×640px input resolution
- **Faster R-CNN**: ResNet-50 backbone with FPN, 800×800px input resolution

### Optimization Techniques
- Data augmentation: Geometric transformations, color modifications, mixing techniques (Mosaic, MixUp)
- Class weighting: Inversely proportional to class frequency
- Learning rate scheduling: Warm-up phase followed by cosine decay

## 📊 Results

YOLOv8 clearly demonstrated superior performance:

| Metric | YOLOv8 | Faster R-CNN |
|--------|--------|-------------|
| mAP50 | 0.919 | 0.82* |
| mAP50-95 | 0.659 | 0.51* |
| Precision | 0.909 | - |
| Recall | 0.864 | - |

*Faster R-CNN results were limited by training constraints (only 12 epochs completed)

### Performance by Class (YOLOv8)

| Class | AP50 | AP50-95 | Precision | Recall |
|-------|------|---------|-----------|--------|
| Car | 0.966 | 0.763 | 0.916 | 0.931 |
| Truck | 0.975 | 0.781 | 0.941 | 0.966 |
| Van | 0.968 | 0.756 | 0.939 | 0.928 |
| Tram | 0.965 | 0.735 | 0.921 | 0.927 |
| Misc | 0.920 | 0.647 | 0.856 | 0.902 |
| Cyclist | 0.850 | 0.536 | 0.875 | 0.783 |
| Pedestrian | 0.831 | 0.471 | 0.894 | 0.732 |
| Person_sitting | 0.877 | 0.586 | 0.932 | 0.746 |

## 📁 Project Structure
```
kitti-object-detection/
├── models/
│   ├── faster_rcnn.py        # Faster R-CNN implementation
│   └── yolov8_advanced.py    # YOLOv8 implementation
├── video_testing/
│   ├── traffic_results_yolov8m.mp4  # YOLOv8 video demonstration
│   ├── traffic.mp4                  # Source video
│   └── video_test_yolov8m.py        # Video testing script
├── data_exploration.ipynb    # Dataset exploration and analysis
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

## 🔧 Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```
### Training Models
```bash
python models/yolov8_advanced.py --train
python models/faster_rcnn.py --train  # Note: Requires significant GPU memory
```
### Testing on Video
```bash
python video_testing/video_test_yolov8m.py --input path/to/video --output results.mp4
```
## 🔮 Future Work

- Extended training for Faster R-CNN (40+ epochs)
- Exploration of hybrid architectures combining single-stage and two-stage benefits
- Evaluation of model robustness across various environmental conditions

## Acknowledgments

- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/) - for providing the dataset
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - for the detection framework
- [Weights & Biases](https://wandb.ai/) - for experiment tracking
- [Hugging Face](https://huggingface.co/) - for model hosting
