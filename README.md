

---
title: AuthentiVision
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AuthentiVision: A Deep Learning-Based Framework for Detecting AI-Generated Content in Digital Marketing

AuthentiVision is an end-to-end Deep Learning and Explainable AI (XAI) framework engineered to detect AI-generated and synthetically manipulated imagery across digital marketing campaigns. Powered by **TensorFlow**, **MobileNetV2 Transfer Learning**, and **Grad-CAM (Gradient-weighted Class Activation Mapping)**, AuthentiVision provides transparent confidence scores and spatial heatmaps to protect brand trust and compliance.

---

## 🌟 Key Features

1. **AI Content Detection**: Fast, lightweight classification using MobileNetV2 (`authentivision_mobilenetv2.h5`) fine-tuned for Real vs. AI-Generated imagery.
2. **Explainable AI (Grad-CAM)**: Visual attention heatmaps computed on layer `out_relu` (1280 channels) highlighting generative diffusion artifacts, edge anomalies, and synthetic textures.
3. **Marketing Campaign Dashboard**: Interactive Flask web interface supporting drag-and-drop single asset verification and batch campaign scanning (multi-image or ZIP archives).
4. **Automated Authenticity Reports**: One-click generation of executive **PDF reports** (via ReportLab) and **CSV audit logs** tracking campaign threat metrics and synthetic media ratios.
5. **REST API**: Programmatic API endpoint (`/api/scan`) for headless integration into marketing CMS or ad-tech pipelines.
6. **Training Pipeline**: Ready-to-run scripts for local or Google Colab training on balanced CIFAKE splits (`CIFAKE_Train50K_Test20K.zip`).

---

## 🏗️ System Architecture

```text
Input Marketing Image (224 x 224 x 3)
         │
         ▼
MobileNetV2 Backbone (TensorFlow 2.x, Pretrained ImageNet)
         │
         ├──► out_relu Layer (1280 channels) ──► Grad-CAM Engine ──► Visual Heatmap Overlay
         │
         ▼
Global Average Pooling 2D + Dropout(0.3)
         │
         ▼
Dense(1, Sigmoid)
         │
         ▼
Confidence Probabilities P(AI-Generated) & P(Real)
```

---

## 📁 Project Structure

```text
authentivision/
├── app.py                                  # Flask Application & Routing Engine
├── requirements.txt                        # Python Dependencies
├── README.md                               # Complete Documentation
├── run_dashboard.sh                        # Linux/macOS Application Launcher
├── run_dashboard.bat                       # Windows Application Launcher
├── AuthentiVision_MobileNetV2_Training.ipynb # Google Colab Ready-to-Run Training Notebook
├── dataset/                                # CIFAKE Dataset (Train 50K / Test 20K)
│   ├── train/ (REAL: 25K, FAKE: 25K)
│   └── test/  (REAL: 10K, FAKE: 10K)
├── models/
│   ├── builder.py                          # MobileNetV2 Architecture & Weight Manager
│   ├── authentivision_mobilenetv2.h5       # Production Trained Model (HDF5 format)
│   └── evaluation_report.json              # 20,000 Test Set Performance Metrics
├── src/
│   ├── __init__.py
│   ├── inference.py                        # Image Preprocessing & Model Prediction
│   ├── gradcam.py                          # TensorFlow Grad-CAM & OpenCV Overlay Generator
│   ├── batch_scanner.py                    # Campaign Batch Scanner & ZIP Extraction
│   ├── report_generator.py                 # PDF & CSV Executive Audit Generator
│   └── train.py                            # Transfer Learning & Evaluation Pipeline
├── sample_data/                            # Preloaded Test Assets (Authentic & AI Samples)
├── static/
│   ├── css/custom.css                      # Modern UI Theme & Metrics Styling
│   ├── js/main.js                          # Drag-and-drop, Chart.js & Modals
│   ├── uploads/                            # Temporary Asset & Heatmap Storage
│   └── reports/                            # Generated PDF & CSV Audit Reports
├── templates/
│   ├── base.html                           # Navigation & Global Layout
│   ├── index.html                          # Dashboard Home & Quick Scan Dropzone
│   ├── single_scan.html                    # Single Image Inspection & Grad-CAM Studio
│   ├── batch_scan.html                     # Batch Campaign Scanner & Metrics Table
│   ├── report.html                         # Formal Executive Report View
│   └── model_info.html                     # Architecture Specifications & XAI Formulations
└── tests/
    └── test_system.py                      # 8-Point End-to-End Test Suite
```

---

## 🚀 Quickstart Guide

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Flask Web Application

**On macOS / Linux:**
```bash
./run_dashboard.sh
```

**Or directly with Python:**
```bash
python app.py
```

**On Windows:**
```cmd
run_dashboard.bat
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🖥️ Web Dashboard Workflow

1. **Dashboard Home (`/`)**:
   - High-level KPI status and framework capabilities.
   - Quick-scan dropzone for instantaneous single-file verification.
   - Quick-test buttons to scan preloaded demo assets with 1 click.
2. **Single Scan Studio (`/scan`)**:
   - Upload any image (`.jpg`, `.png`, `.webp`, `.jpeg`).
   - View classification verdict (Authentic vs. AI-Generated) and confidence meter.
   - Explore the **Grad-CAM Visualizer** across 3 views: *Composite Overlay*, *Raw Heatmap*, and *Original Asset*.
3. **Batch Campaign Scanner (`/batch`)**:
   - Upload multiple image creatives or a single `.zip` file.
   - Visualizes campaign risk level: **Low** (<15% AI), **Moderate** (15-40% AI), or **High** (>40% AI).
   - Interactive Chart.js breakdown and searchable data table with modal Grad-CAM previews.
4. **Audit Reports (`/report/<campaign_id>`)**:
   - Executive-ready formal audit presentation.
   - One-click buttons to download high-resolution **PDF reports** and **CSV audit logs**.

---

## 🎯 Model Training (CIFAKE Dataset)

### Dataset Configuration
The dataset archive `CIFAKE_Train50K_Test20K.zip` is partitioned into:
- **Training Subset:** 40,000 images (20,000 REAL + 20,000 FAKE)
- **Validation Subset:** 10,000 images (5,000 REAL + 5,000 FAKE)
- **Evaluation Test Set:** 20,000 images (10,000 REAL + 10,000 FAKE)

### Local Training
```bash
python src/train.py --data_dir dataset --epochs 5 --batch_size 64 --lr 0.001
```
*Output saved to:* `models/authentivision_mobilenetv2.h5`

### Google Colab GPU Training
Open and run **`AuthentiVision_MobileNetV2_Training.ipynb`** in Google Colab with GPU runtime for ~90-second training.

---

## 🔌 REST API Documentation

### Endpoint: `POST /api/scan`
Classifies an uploaded image and returns prediction scores and Grad-CAM artifact URLs.

#### Request (Multipart Form):
```bash
curl -X POST http://127.0.0.1:5000/api/scan -F "image=@sample_data/sample_authentic_camera.jpg"
```

#### JSON Response:
```json
{
  "status": "success",
  "prediction": "Real",
  "verdict": "Authentic Digital Asset",
  "confidence": 92.4,
  "real_probability": 92.4,
  "ai_probability": 7.6,
  "risk_level": "Low Risk",
  "explanation": "The asset exhibits organic photographic qualities with 92.4% authenticity confidence...",
  "overlay_url": "/static/uploads/api_1731911234_overlay.jpg",
  "heatmap_url": "/static/uploads/api_1731911234_heatmap.jpg"
}
```

---

## 🧪 System Verification

Run the full automated test suite covering all 8 endpoints and pipelines:
```bash
python tests/test_system.py
```
