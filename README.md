# Ganoderma-Adversarial-YOLOv11
Repository for the paper: **Adversarial Robustness of Ganoderma Detection in Oil Palm: A Cross-Paradigm Comparison of YOLOv11 and ResNet50 under FGSM Attacks.**

This repository contains the official implementation, replication scripts, and reproducibility assets for our research on securing AI-driven vision systems in precision agriculture against adversarial vulnerabilities.

## 🛡️ Threat Model & Evaluation Scope
This project evaluates the adversarial vulnerability of deep learning models used for detecting *Ganoderma boninense* fungus. We compare the decay behaviors of two distinct computer vision paradigms:
*   **YOLOv11** (Object Detection) - Evaluated using Mean Average Precision (mAP@0.5).
*   **ResNet50** (Image Classification) - Evaluated using absolute classification Accuracy.

**Methodological Note:** The current implementation utilizes the **Fast Gradient Sign Method (FGSM)**. Rather than simulating a real-world cyberattack, FGSM is applied as a controlled, white-box stress test to establish a baseline for adversarial vulnerability. *More complex iterative attacks, such as Projected Gradient Descent (PGD), are scoped for future work and are isolated from the primary evaluation.*

## 📁 Repository Structure
├── dataset_sample/      # Contains a small subset of annotated oil palm images and DATA_POLICY.txt.
├── models/              # Directory framework for trained weights.
├── notebooks/           # Google Colab notebooks containing the experimental pipeline.
│   ├── 01_Baseline_Training.py
│   ├── 02_FGSM_Attack_Simulation.py
│   ├── 03_Adversarial_Training.py
│   └── generate_figure4.py  # Generates the side-by-side metric separation subplot
├── future_work/         # Experimental iterative attacks (contains isolated PGD implementation)
├── requirements.txt     # Python dependencies required to replicate the environment.
└── README.md            # Repository documentation and setup instructions.

## 💾 Trained Model Weights Download Links
To replicate the evaluation environment, download the files manually and place them into the `models/` directory:
* 📥 **YOLOv11 Nano Baseline Weights (`yolov11_baseline.pt`):** `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/yolov11_baseline_gpu.pt`
* 📥 **Proposed Adversarially Trained YOLOv11 Weights (`yolov11_defended.pt`):** `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/yolov11_defended_gpu.pt`

## 🛠️ Installation & Setup
1. **Clone the Repository:**
```bash
git clone [https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11.git](https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11.git)
cd Ganoderma-Adversarial-YOLOv11
