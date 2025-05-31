# Chest X-Ray Pneumonia Classification

This project implements a deep learning pipeline to classify chest X-ray images for pneumonia detection using PyTorch. It supports multiple image preprocessing techniques and model architectures.

---

## Features

- Dataset loading and preprocessing with grayscale image handling.
- Three transformation modes:  
  - **scale:** Simple scaling and resizing.  
  - **zscore:** Standard normalization using dataset mean and std.  
  - **normalize:** Custom normalization involving cropping based on cumulative distribution function (CDF) of grayscale intensities.
- Support for three model architectures:  
  - Custom CNN from scratch  
  - Pretrained MobileNetV2 adapted for grayscale input  
  - Pretrained EfficientNet-B0 adapted for grayscale input
- Training and evaluation with accuracy, loss, classification report, and ROC curve metrics.
- Logging outputs to timestamped log files.
- Saving trained model weights and performance plots.

---

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- opencv-python
- pillow
- torchviz (optional, for model visualization)

Install dependencies using:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn opencv-python pillow torchviz
```

## Dataset

chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── val/  (optional)

data_dir = r"C:\Users\Study\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray"

## Configuration

| Parameter        | Description                                                  | Default   |
| ---------------- | ------------------------------------------------------------ | --------- |
| `transform_mode` | Preprocessing mode: "scale", "zscore", or "normalize"        | `"scale"` |
| `model_name`     | Model architecture: "cnn", "mobilenetv2", "efficientnet\_b0" | `"cnn"`   |
| `img_height`     | Image height after resizing                                  | 256       |
| `img_width`      | Image width after resizing                                   | 256       |
| `batch_size`     | Batch size for training and testing                          | 1 or 32   |
| `epochs`         | Number of training epochs                                    | 1         |
| `num_classes`    | Number of output classes (e.g., 2 for pneumonia detection)   | 2         |

## Usage

Set up dataset path and parameters.

Choose the preprocessing mode and model architecture.

Run the script:

bash
Copy
Edit
python train.py
(Replace train.py with your actual script filename.)

The training logs will be saved automatically to a file named like log_YYYY-MM-DD_HH-MM-SS.txt.

After training, the following are saved:

Model weights: xray_classifier.pth

Performance plots: model_metrics.png

## Code Overview

Data Loading and Preprocessing
Uses torchvision.datasets.ImageFolder for standard loading.

Calculates dataset mean and std for normalization when needed.

Custom dataset class for advanced cropping and normalization (transform_mode "normalize").

Models
CNN: Custom 3-layer convolutional network for grayscale input.

MobileNetV2: Pretrained on ImageNet, adapted for single-channel input.

EfficientNet-B0: Pretrained model adapted for grayscale images.

Training & Evaluation
Optimizer: Adam with learning rate scheduler.

Loss function: CrossEntropyLoss.

Tracks training and validation accuracy and loss.

Generates classification report with precision, recall, and F1-score.

Plots accuracy, loss, classification metrics, and ROC curve.

## Output

Log file: Contains console output saved with timestamp.

Model weights: Saved as xray_classifier.pth.

Metrics plot: Saved as model_metrics.png.

## Notes

The current script runs for 1 epoch by default; increase epochs for better performance.

Batch size is set low for compatibility; adjust depending on your hardware.

Windows users should keep num_workers=0 to avoid multiprocessing issues.

## License
This project is open source and free to use.

## Acknowledgments
Dataset from Paul Mooney's Chest X-Ray Pneumonia Dataset.

Pretrained models from torchvision.