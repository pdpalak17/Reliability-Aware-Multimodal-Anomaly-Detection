---
title: Reliability-Aware Contextual Multimodal Anomaly Detection
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app_gradio.py
pinned: false
---

# 🛡️ Reliability-Aware Contextual Multimodal Anomaly Detection System

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://share.streamlit.io)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-ZeroGPU-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Palakdwivedi1706/Multimodal-Anomaly-Detection)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Skeleton%20Pose-orange?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)

---

## 📌 Project Overview

This repository contains the official implementation of **Reliability-Aware Contextual Multimodal Anomaly Detection**, developed at **Woxsen University, School of Technology**.

Traditional single-modality surveillance systems suffer from high false positive rates under sensor degradation (e.g., face occlusion, dark illumination, camera noise, erratic motion). This framework introduces a **dynamic reliability-weighted attention fusion architecture** that dynamically adjusts modality weights $\alpha_m$ based on real-time sensor confidence scores $w_m \in [0, 1]$.

```
┌─────────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
│  Facial Expression CNN  │    │ Body Pose Network (99D)  │    │ Video CNN-LSTM (128D)   │    │ Context Metadata (32D)  │
│  (64D Emotion Vectors)  │    │  (MediaPipe Keypoints)   │    │ (Temporal Motion Vector)│    │  (Zone, Hour, Crowd)    │
└────────────┬────────────┘    └────────────┬─────────────┘    └────────────┬────────────┘    └────────────┬────────────┘
             │                              │                               │                              │
             └──────────────────────────────┼───────────────────────────────┴──────────────────────────────┘
                                            ▼
                           ┌──────────────────────────────────┐
                           │ Reliability Attention Allocation │
                           │   w_f · w_p · w_v · w_c Fusion   │
                           └────────────────┬─────────────────┘
                                            ▼
                           ┌──────────────────────────────────┐
                           │   Anomaly Probability & XAI RAG  │
                           │   Plain-Language Alert Engine    │
                           └──────────────────────────────────┘
```

---

## ✨ Key Technical Features

1. **🤖 Computer Vision Auto-Detection Engine**:
   - Integrated OpenCV multi-scale Haar Cascade face detection with histogram equalization and Non-Maximum Suppression (NMS) to automatically count faces and align bounding boxes.

2. **🦴 33-Landmark MediaPipe Skeleton Overlay**:
   - Renders 7 distinct action pose archetypes (*Standing, Gesturing, Running, Crouching, Falling, Aggressive, Defensive*).

3. **😊 7-Class Facial Emotion Recognition**:
   - Real-time 7-class facial emotion classification (`Neutral`, `Happy`, `Angry`, `Fear`, `Sad`, `Surprise`, `Disgust`) rendered directly above face bounding boxes.

4. **⚡ Fully Reactive Dashboard**:
   - Re-evaluates risk probabilities, reliability scores, modality attention progress bars, and RAG plain-language explanations on **every click, slider adjustment, or video frame scrub step**.

5. **🤖 Explainable AI (SHAP) & RAG Precedent Retrieval**:
   - Generates human-readable natural language alerts grounded in historical incident precedents retrieved from a vector knowledge base.

---

## 🌿 Repository Branch Structure

| Branch Name | Primary Purpose & Contents | Target Platform |
| :--- | :--- | :--- |
| **`main`** | Full-featured Streamlit production dashboard (`app.py`), reactive UI, multi-person video processor. | [Streamlit Community Cloud](https://share.streamlit.io) |
| **`gradio-zerogpu`** | Gradio interface (`app_gradio.py`), `@spaces.GPU` ZeroGPU decorators, pinned dependencies. | [Hugging Face Spaces](https://huggingface.co/spaces/Palakdwivedi1706/Multimodal-Anomaly-Detection) |
| **`feature/multimodal-fusion-rag`** | Neural network branch architectures, fusion network, SHAP importance & RAG alert engine (`models/`). | Feature Testing |
| **`feature/computer-vision-processor`** | OpenCV multi-scale face detector, NMS bounding box filter, MediaPipe 33-landmark skeleton renderer (`utils/`). | Feature Testing |
| **`docker-deployment`** | Containerization configuration (`Dockerfile`, `.dockerignore`) for cloud deployments. | Render / Railway / AWS ECR |

---

## 🚀 Quickstart & Local Setup

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/pdpalak17/Reliability-Aware-Multimodal-Anomaly-Detection.git
cd Reliability-Aware-Multimodal-Anomaly-Detection
pip install -r requirements.txt
```

### 2. Run Local Streamlit Dashboard
```bash
python -m streamlit run app.py
```
*Open your browser at `http://localhost:8501` to view the live dashboard.*

### 3. Run Local Gradio Interface
```bash
python app_gradio.py
```
*Open your browser at `http://localhost:7860`.*

### 4. Run Automated Unit Tests
```bash
python tests/test_pipeline.py
```

---

## 📊 Benchmark Evaluation Metrics

Evaluated across three benchmark datasets (**Multimodal Emotion**, **MPII Human Pose**, **UCF-Crime**):

| Modality Configuration | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Single Modality (Facial CNN) | 81.2% | 79.5% | 80.1% | 79.8% | 0.84 |
| Single Modality (MPII Pose) | 84.6% | 83.2% | 84.0% | 83.6% | 0.87 |
| Unweighted Feature Concatenation | 87.5% | 86.1% | 86.9% | 86.5% | 0.90 |
| **Proposed Reliability-Aware Fusion** | **94.6%** | **93.8%** | **94.1%** | **93.9%** | **0.95** |

---

## 📂 Project Organization

```text
Reliability-Aware-Multimodal-Anomaly-Detection/
├── app.py                      # Main Streamlit Dashboard Application
├── app_gradio.py               # Gradio Application for Hugging Face ZeroGPU
├── Dockerfile                  # Container Deployment Dockerfile
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
├── data/                       # Datasets & Knowledge Base
│   ├── dataset_loaders.py      # Multimodal Data Loader Pipeline
│   └── incident_kb.json        # RAG Knowledge Base Precedents
├── models/                     # Deep Learning Neural Networks
│   ├── face_branch.py          # Facial Expression CNN
│   ├── pose_branch.py          # Body Pose Network
│   ├── video_branch.py         # CNN-LSTM Temporal Motion Network
│   ├── context_branch.py       # Context Metadata MLP
│   ├── attention_fusion.py     # Reliability Attention Fusion Network
│   └── xai_rag_engine.py       # SHAP Explainability & RAG Alert Engine
├── saved_models/               # Pretrained Model Weights (.npz)
├── training/                   # Model Training Scripts
├── utils/                      # Computer Vision Utilities
│   ├── video_processor.py      # OpenCV Face Detection & MediaPipe Skeleton
│   └── metrics.py              # Performance Metrics Calculator
├── tests/                      # Automated Unit Test Suite
│   └── test_pipeline.py        # 7-Step Pipeline Unit Tests
└── docs/                       # Project Documentation & Presentations
```

---

## 👥 Authors & Acknowledgments

- **Authors:** Palak Dwivedi, T. Sri Vaishnavi, Ojashwini Dubey, Spoorthi Reddy, Tiasha Roy
- **Supervisor:** Dr. Uday Chandra
- **Institution:** Woxsen University, School of Technology

---

## 📜 License & Citation

This project is licensed under the MIT License. If you use this repository in your research, please cite:

```bibtex
@article{dwivedi2026reliability,
  title={Reliability-Aware Contextual Multimodal Anomaly Detection in Surveillance Streams},
  author={Dwivedi, Palak and Vaishnavi, T. Sri and Dubey, Ojashwini and Reddy, Spoorthi and Roy, Tiasha and Chandra, Uday},
  journal={Woxsen University School of Technology Research Proceedings},
  year={2026}
}
```
