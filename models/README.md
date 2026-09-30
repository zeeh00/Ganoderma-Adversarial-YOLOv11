# Models Directory

This directory is designated for holding the trained weights of the deep learning architectures utilized throughout this research.

## 📁 Pre-Trained Model Weights
To run inference, attack simulations, or evaluations without retraining the YOLO architectures from scratch, download the pre-trained weights and place them directly into this folder:

1. **`yolov11_baseline_gpu.pt`**
   * *Description:* Standard YOLOv11 Nano object detection architecture optimized on the pristine (clean) oil palm dataset.
   * *Download Link:* `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/yolov11_baseline_gpu.pt`

2. **`yolov11_defended_gpu.pt`**
   * *Description:* The proposed secure YOLOv11 Nano framework reinforced via robust Adversarial Training.
   * *Download Link:* `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/yolov11_defended_gpu.pt`

## ℹ️ ResNet-50 Baseline Note
* **`resnet50_baseline_gpu.pth`:** ResNet-50 utilizes the standard `torchvision.models.resnet50(weights='DEFAULT')` backbone and is trained and evaluated directly on-the-fly via `notebooks/01_Baseline_Training.py`. Running that script will generate and save the checkpoint locally into this directory upon completion.

## ⚠️ Important Note for Git Users
Actual binary weight parameters (`.pt` and `.pth` files) are structurally excluded from direct Git tracking to maintain repository lightweight optimization and prevent version control bloat. Ensure that the filenames of the downloaded or trained models match these references exactly before executing any scripts within the `notebooks/` directory.
