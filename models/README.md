# Models Directory

This directory is designated for holding the trained weights of the deep learning architectures utilized throughout this research.

## 📁 Required Model Files
To successfully execute the replication notebooks and run independent inference pipelines, you must manually download the compiled network weights and place them directly into this folder:

1. **`yolov11_baseline_gpu.pt`** 
   * *Description:* Standard YOLOv11 Nano object detection architecture optimized strictly under pristine (clean) environmental data.
   * *Download Link:* `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/yolov11_baseline_gpu.pt`

2. **`resnet50_baseline_gpu.pth`** 
   * *Description:* Standard ResNet50 image classification architecture trained on the clean Ganoderma dataset to serve as the comparative baseline.
   * *Download Link:* `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/resnet50_baseline_gpu.pth`

3. **`yolov11_defended_gpu.pt`** 
   * *Description:* The proposed secure YOLOv11 Nano framework reinforced via robust Adversarial Training.
   * *Download Link:* `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/yolov11_defended_gpu.pt`

## ⚠️ Important Note for Git Users
Actual binary weight parameters (`.pt` and `.pth` files) are structurally excluded from direct Git tracking to maintain repository lightweight optimization and prevent version control bloat. Ensure that the filenames of the downloaded models match these references exactly before executing any scripts within the `notebooks/` directory.
