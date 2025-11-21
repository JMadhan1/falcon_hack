# Duality AI Space Station Challenge #2 - Safety Object Detection

This repository contains the solution for the Duality AI Space Station Challenge #2. The goal is to train a YOLOv8 model to detect 7 safety objects in a simulated space station environment.

## 📊 Dataset
- **Source:** Duality AI FalconEditor (Synthetic Data)
- **Classes (7):** `OxygenTank`, `NitrogenTank`, `FirstAidBox`, `FireAlarm`, `SafetySwitchPanel`, `EmergencyPhone`, `FireExtinguisher`
- **Split:**
    - Train: 1,767 images
    - Val: 336 images
    - Test: 1,408 images

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd falconhack
   ```

2. **Create Environment:**
   ```bash
   conda create -n EDU python=3.10 -y
   conda activate EDU
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage Instructions

### 1. Train the Model
To reproduce the training results:
```bash
python train.py
```
*This trains YOLOv8n for 100 epochs and saves the best model to `runs/detect/train/weights/best.pt`.*

### 2. Evaluate Performance
To generate validation metrics (mAP, Confusion Matrix):
```bash
python validate.py
```

### 3. Run Inference (Test Set)
To generate predictions on the test set:
```bash
python predict.py
```
*Results will be saved to `runs/detect/test/` including `predictions.csv`.*

### 4. Failure Analysis
To generate a report of failure cases (False Positives/Negatives):
```bash
python failure_analysis.py
```
*Images highlighting failures will be saved to `runs/detect/failure_analysis/`.*

## 📈 Results
- **mAP@0.5:** 0.714
- **mAP@0.5-95:** 0.552

## 📂 Repository Structure
```
falconhack/
├── dataset/               # Train/Val/Test images and labels
├── runs/                  # Training logs, weights, and visualizations
├── train.py               # Training script
├── validate.py            # Validation script
├── predict.py             # Inference script (renamed from test.py)
├── failure_analysis.py    # Script to analyze model failures
├── data.yaml              # YOLO dataset config
├── HACKATHON_REPORT.md    # Detailed project report
└── README.md              # This file
```

## 📝 License
This project is part of the Duality AI Hackathon.
