# Ganoderma-Adversarial-YOLOv11
Repository for the paper: Enhancing Cyber Security in Precision Agriculture: Defending Ganoderma Detection Mobile Robots Against Adversarial Attacks.

This repository contains the official implementation, replication scripts, and reproducibility assets for our research on securing AI-driven vision systems in precision agriculture against adversarial vulnerabilities.


## 📁 Repository Structure
├── dataset_sample/      # Contains a small subset of annotated oil palm images and DATA_POLICY.txt.
├── models/              # Directory framework for trained weights.
├── notebooks/           # Google Colab notebooks containing the experimental pipeline.
│   ├── 01_Baseline_Training.py
│   ├── 02_FGSM_Attack_Simulation.py
│   └── 03_Adversarial_Training.py
├── requirements.txt     # Python dependencies required to replicate the environment.
└── README.md            # Repository documentation and setup instructions.


## 💾 Trained Model Weights Download Links
To replicate the evaluation environment, download the files manually and place them into the `models/` directory:
* 📥 **YOLOv11 Nano Baseline Weights (`yolov11_baseline.pt`):** `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/yolov11_baseline_gpu.pt`
* 📥 **Proposed Adversarially Trained YOLOv11 Weights (`yolov11_defended.pt`):** `https://github.com/zeeh00/Ganoderma-Adversarial-YOLOv11/blob/main/models/yolov11_defended_gpu.pt`


## 🛠️ Installation & Setup
1. **Clone the Repository:**
   bash
   git clone [https://github.com/USERNAME/REPOSITORY-NAME.git](https://github.com/USERNAME/REPOSITORY-NAME.git)
   cd REPOSITORY-NAME

2. **Install Dependencies:**
bash
pip install -r requirements.txt


## 🚀 How to Run the Experiments
The core replication workflow is segmented logically across three execution steps inside the `notebooks/` folder:

1. **Baseline Training (`notebooks/01_Baseline_Training.py`):** Handles environment initialization, Roboflow dataset ingestion, and baseline optimization of YOLOv11 Nano and ResNet50 models under pristine (clean) conditions using GPU acceleration (`device=0`).
2. **Adversarial Attack Simulation (`notebooks/02_FGSM_Attack_Simulation.py`):** Implements the white-box Fast Gradient Sign Method (FGSM) attack pipeline, evaluating baseline model degradation across variable epsilon bounds (ε ∈ {0.01, 0.05, 0.10, 0.30}).
3. **Robust Evaluation & Defense (`notebooks/03_Adversarial_Training.py`):** Retrains the secure YOLOv11 model framework using robust adversarial training, logs final defense recovery metrics, and automatically generates publication assets (**Figure 5** and **Figure 6**).


## 📊 Key Empirical Findings
* **Baseline Accuracy:** The standard YOLOv11 framework achieves a Mean Average Precision (mAP@0.5) of 92.30% on clean data.
* **Vulnerability Collapse:** Under a maximum adversarial pixel manipulation (ε = 0.30), the standard model's classification rate drops catastrophically to 17.02%.
* **Defense Recovery:** The proposed Adversarially Trained YOLOv11 framework effectively builds structural resilience against gradient disruptions, maintaining a robust final accuracy of 72.34% under the most severe noise levels.

## ⚖️ Data Policy
Please refer to `dataset_sample/DATA_POLICY.txt` for legal guidelines and disclosures regarding the proprietary, non-distributable nature of our commercial oil palm plantation dataset gathered in the North Sumatra region, Indonesia.
