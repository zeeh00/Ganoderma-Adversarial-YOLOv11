## Threat Model Scope
This repository evaluates the adversarial robustness of Ganoderma detection models. The current implementation utilizes the **Fast Gradient Sign Method (FGSM)** as a controlled, white-box stress test to establish a baseline for adversarial vulnerability. 

*Note: More complex iterative attacks, such as Projected Gradient Descent (PGD), are scoped for future work and are not part of the primary comparative baseline in this phase.*

## Evaluation Metrics Disclaimer
This comparative analysis evaluates two distinct paradigms:
* **YOLOv11 (Object Detection):** Evaluated using Mean Average Precision (mAP@0.5).
* **ResNet50 (Image Classification):** Evaluated using absolute classification Accuracy.
Because these metrics are mathematically distinct, the degradation trends are presented to illustrate relative model decay under perturbation, rather than as a direct 1:1 metric comparison.

## Data Availability
The raw dataset of Ganoderma-infected oil palms used to train these models is not publicly available due to proprietary agreements with plantation business partners in North Sumatra. This repository provides the supporting experimental code, attack frameworks, and evaluation scripts used to generate the results.
