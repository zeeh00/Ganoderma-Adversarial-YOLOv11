# =====================================================================
# STEP 1: INSTALL AND IMPORT REQUIRED LIBRARIES
# =====================================================================
# Install the official Ultralytics framework for YOLOv11 engine
!pip install ultralytics -q
!pip install roboflow -q

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from ultralytics import YOLO

# Validate CUDA hardware acceleration availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("==============================================================")
print(f"[✓] Execution Device: {str(device).upper()}")
print(f"[✓] PyTorch Version : {torch.__version__}")
print("==============================================================")


# =====================================================================
# STEP 2: REPRODUCIBLE DATASET DOWNLOADING
# =====================================================================
# Ingest the annotated oil palm tree image dataset via the Roboflow API
from roboflow import Roboflow

# Initialize Roboflow client (Replace with your repository specific keys)
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("your-workspace").project("oil-palm-ganoderma")
dataset = project.version(2).download("yolov8")

# Establish core directory path variables
DATASET_DIR = dataset.location
print("--------------------------------------------------------------")
print(f"[✓] Dataset repository securely mounted at: {DATASET_DIR}")
print("--------------------------------------------------------------")


# =====================================================================
# STEP 3: BASELINE OBJECT DETECTION MODEL TRAINING
# =====================================================================
print("\n=== STARTING BASELINE YOLOV11 NANO OPTIMIZATION VIA CUDA ===")

# 1. Instantiate the baseline architectural weights of YOLOv11 Nano [cite: 5, 21]
model_yolo = YOLO('yolo11n.pt')

# 2. Execute training pipeline on the clean agricultural dataset [cite: 91]
# Hyperparameters are tightly bound to the empirical study configuration [cite: 65, 92, 93]
results_yolo = model_yolo.train(
    data=os.path.join(DATASET_DIR, "data.yaml"), # Ingest dataset structures [cite: 54]
    epochs=50,                                   # Standard iteration depth [cite: 92, 93]
    imgsz=640,                                   # Pixel resolution dimension [cite: 65, 74]
    batch=16,                                    # Balanced mini-batch size for GPU VRAM stability
    name="yolov11_baseline_gpu",                 # Directory tag for output artifacts [cite: 54]
    device=0,                                    # Enforce training via CUDA:0 GPU 
    exist_ok=True                                # Prevent nested subfolder generation
)

print("\n[✓] Object Detector Optimization Completed.")
print(f"[✓] Best baseline weights compiled at: /content/runs/detect/yolov11_baseline_gpu/weights/best.pt")


# =====================================================================
# STEP 4: CUSTOM PRE-PROCESSING & PYTORCH PIPELINE FOR RESNET50
# =====================================================================
# Define standardized ImageNet validation transforms for the image classification model [cite: 32]
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class OilPalmBinaryDataset(Dataset):
    """
    Custom PyTorch wrapper mapping object detection image targets 
    into absolute binary classification categories for structural validation[cite: 105].
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Ground truth mapping: file strings containing 'ganoderma' are labeled 1, else 0 [cite: 106, 107]
        label = 1 if "ganoderma" in self.img_files[idx].lower() else 0
        
        if self.transform:
            image = self.transform(image)
        return image, label

# Construct highly optimized DataLoader pipes
train_images_path = os.path.join(DATASET_DIR, 'train/images')
train_dataset_resnet = OilPalmBinaryDataset(img_dir=train_images_path, transform=transform_resnet)
train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=16, shuffle=True, num_workers=2)

print("--------------------------------------------------------------")
print(f"[✓] PyTorch Dataloader successfully mapped {len(train_dataset_resnet)} training assets.")
print("--------------------------------------------------------------")


# =====================================================================
# STEP 5: RESNET50 COMPARATIVE CLASSIFIER OPTIMIZATION
# =====================================================================
print("\n=== STARTING BENCHMARK RESNET50 PIPELINE VIA CUDA ===")

# 1. Ingest baseline deep residual architecture utilizing standard weights [cite: 32]
model_resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

# 2. Reshape the final Fully Connected (FC) linear layer for binary evaluation [cite: 105]
num_features = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_features, 2) # Output layer mapped to normal vs infected targets [cite: 106, 107]
model_resnet = model_resnet.to(device)

# 3. Formulate loss parameters and Adam gradient optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_resnet.parameters(), lr=0.001)

# 4. Standard PyTorch Training Loop
epochs_resnet = 10 
model_resnet.train()

for epoch in range(epochs_resnet):
    running_loss = 0.0
    for images, labels in train_loader_resnet:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model_resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader_resnet.dataset)
    print(f"   Epoch [{epoch+1}/{epochs_resnet}] -> CrossEntropy Loss: {epoch_loss:.4f}")

# 5. Export binary classification weights to storage
os.makedirs('/content/models_saved', exist_ok=True)
torch.save(model_resnet.state_dict(), '/content/models_saved/resnet50_baseline.pth')

print("\n==============================================================")
print("[✓] Baseline ResNet50 Classifier Training Process Finalized.")
print("[✓] Model saved successfully as: /content/models_saved/resnet50_baseline.pth")
print("==============================================================")
