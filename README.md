---
title: Reliability-Aware Contextual Multimodal Anomaly Detection
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# 🛡️ Reliability-Aware Contextual Multimodal Anomaly Detection

**Submitted by:** Palak Dwivedi, T. Sri Vaishnavi, Ojashwini Dubey, Spoorthi Reddy, Tiasha Roy  
**Institution:** Woxsen University, School of Technology  
**Supervisor:** Dr. Uday Chandra  

---

## 📌 Project Overview

This project implements a **Reliability-Aware Contextual Multimodal Anomaly Detection** framework for video surveillance and smart monitoring. The framework jointly reasons over four distinct modalities:
1. **Facial Expression Branch**: CNN classifier trained on the *Multimodal Emotion Recognition Dataset*.
2. **Body Posture Branch**: 33 3D skeletal landmark keypoint network trained on the *MPII Human Pose Dataset*.
3. **Video Dynamics Branch**: CNN + LSTM temporal network trained on the *UCF-Crime Dataset*.
4. **Scene Context Branch**: Feed-Forward Dense Neural Network processing structured scene metadata.

### 🧠 Context-Aware Attention & Explainable AI (XAI)
- **Attention Fusion**: Dynamically allocates Softmax attention weights $[\alpha_{\text{face}}, \alpha_{\text{pose}}, \alpha_{\text{video}}, \alpha_{\text{context}}]$ per frame to remain resilient against face occlusion and poor lighting.
- **XAI & RAG Plain-Language Alerts**: Computes Grad-CAM visual heatmaps, SHAP feature attributions, and retrieves past incident precedents using a RAG vector engine to compose human-readable alert justifications.

---

## 📊 Model Evaluation Summary

| # | Model / Modality Branch | Connected Dataset | Target Task | Accuracy (%) |
|---|---|---|---|---|
| **1** | Facial Expression CNN | Multimodal Emotion Recognition Dataset | 7 Emotion Classes | **70.20%** |
| **2** | Body Pose Network | MPII Human Pose Dataset | 4 Posture Classes | **66.90%** |
| **3** | Video Temporal CNN+LSTM | UCF-Crime Dataset | 5 Crime & Motion Classes | **43.12%** |
| **4** | **Attention Fusion Network** | **Joint Multimodal Fusion** | **5 Anomaly Categories** | **75.12%** |

- **System Mean Reliability Index**: **90.0%**

---

## 🛠️ Installation & Local Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/Reliability-Aware-Multimodal-Anomaly-Detection.git
cd Reliability-Aware-Multimodal-Anomaly-Detection
pip install -r requirements.txt
```

### 2. Train Models & Evaluate Accuracy
```bash
python training/train_facial_model.py
python training/train_pose_model.py
python training/train_temporal_model.py
python training/train_fusion_model.py
python evaluate_all.py
```

### 3. Launch Dashboard Locally
```bash
streamlit run app.py
```

---

## 🐳 Docker Deployment

### Build & Run Container
```bash
docker build -t multimodal-anomaly-detection .
docker run -p 8501:8501 multimodal-anomaly-detection
```

---

## 🤗 Hugging Face Spaces Deployment Instructions

### Option 1: Streamlit SDK (Recommended for `app.py`)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Select **Streamlit** as the SDK.
3. **Important Hardware Setting:** Under Space Settings ➔ Space Hardware, set Hardware to **CPU Basic (Free)**. *(Do NOT select ZeroGPU because ZeroGPU is only supported on Gradio SDK).*
4. Push/Upload this repository:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME
git push hf main
```

### Option 2: Gradio SDK (For `app_gradio.py` with ZeroGPU)
If you want to use ZeroGPU hardware acceleration, deploy `app_gradio.py` with this frontmatter in `README.md`:
```yaml
---
title: Reliability-Aware Multimodal Anomaly Detection
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app_gradio.py
pinned: false
---
```
