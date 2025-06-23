# 🪖 Safety Helmet Detection with YOLOv11

A computer vision project for detecting **safety helmets** in images and videos using **YOLOv11**, combined with tools like **Ultralytics**, **Optuna**, **MLflow**, **FastAPI**, **DVC**, and **Papermill** for a scalable and trackable ML pipeline.

---

## 🧠 Overview

This project fine-tunes a YOLOv11m model to detect safety helmets using a custom dataset. It utilizes Roboflow for data preparation, **Optuna** for automated hyperparameter search, **MLflow** for experiment logging, and **DVC** for data and model versioning. A **FastAPI** endpoint is also available for deployment.

---

## 📁 Project Structure

```
safety_helmet_YOLOV11/
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── dvc.yaml
├── params.yaml
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
├── inference/
│   ├── medium_inference.py
│   └── large_inference.py
├── fastapi_app/
│   └── main.py
├── notebooks/
│   └── analysis.ipynb
├── scripts/
│   ├── train.py
│   └── optuna_search.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧪 Key Features

- ✅ Helmet detection using YOLOv11m
- 📦 Dataset handled via Roboflow (exported in YOLO format)
- 🧠 Automated hyperparameter tuning using Optuna
- 🧪 Experiment tracking using MLflow
- 📂 Reproducible pipelines with DVC
- ⚡ FastAPI app for live inference
- 📊 Jupyter/Papermill notebooks for visual analysis

---

## 🛠 Installation

### 1. Clone the Repository

```
git clone https://github.com/MarwanAbdellah/Safety_helmet_YOLOV11.git
cd Safety_helmet_YOLOV11
```

### 2. Create and Activate a Virtual Environment

```
python -m venv venv
```

- **Windows**:  
```
venv\Scripts\activate
```

- **macOS/Linux**:  
```
source venv/bin/activate
```

### 3. Install Requirements

```
pip install -r requirements.txt
```

---

## 🚀 Running Inference

```
from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/detect/train/weights/best.pt")
results = model("path/to/image.jpg")

img = Image.fromarray(results[0].plot())
img.show()
img.save("prediction.jpg")
```

---

## ⚙️ Training the Model

```
!python scripts/train.py --img 640 --epochs 50 --data dataset/data.yaml --weights yolov11m.pt
```

Or with Optuna tuning:

```
!python scripts/optuna_search.py --trials 30
```

---

## 🌐 FastAPI Inference

Run:

```
cd fastapi_app
uvicorn main:app --reload
```

Then open your browser at:  
**http://127.0.0.1:8000/docs**

---

## 📊 Dataset Source

Dataset will be available on **Roboflow**:  
**URL:** `https://universe.roboflow.com/YOUR_PROJECT_HERE`

Class annotations:

| Class      | Description               |
|------------|---------------------------|
| `helmet`   | Person wearing helmet     |
| `nohelmet` | Person without helmet     |

---

## 📦 Requirements

```
ultralytics
torch
torchvision
optuna
mlflow
fastapi
dvc
pillow
matplotlib
papermill
```

Install using:

```
pip install -r requirements.txt
```

---

## 📌 .gitignore Suggestions

```
__pycache__/
.venv/
*.pt
runs/
*.jpg
*.log
.ipynb_checkpoints/
dataset/
```

---

## 📬 Contact

**Marwan Abdellah**  
📧 [marawan.abdellah0@gmail.com](mailto:marawan.abdellah0@gmail.com)  
🔗 GitHub: [@MarwanAbdellah](https://github.com/MarwanAbdellah)

