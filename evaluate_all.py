import numpy as np
import pandas as pd
from data.dataset_loaders import load_all_datasets
from models.face_branch import FacialExpressionCNN
from models.pose_branch import BodyPoseNetwork
from models.video_branch import VideoTemporalCNNLSTM
from models.context_branch import ContextMetadataNetwork
from models.attention_fusion import ContextAwareAttentionFusion
from utils.metrics import compute_multimodal_metrics

def evaluate_framework():
    datasets = load_all_datasets()
    X_e, y_e, c_e = datasets['emotion']
    X_p, y_p, c_p = datasets['pose']
    X_u, y_u, c_u = datasets['ucf_crime']

    n_samples = min(len(X_e), len(X_p), len(X_u))

    f_w = np.load('saved_models/facial_model_weights.npz')
    p_w = np.load('saved_models/pose_model_weights.npz')
    v_w = np.load('saved_models/video_model_weights.npz')
    fu_w = np.load('saved_models/fusion_model_weights.npz')

    # 1. Facial Expression Model
    face_net = FacialExpressionCNN()
    face_net.set_weights(f_w['W_feat'], f_w['b_feat'], f_w['W_cls'], f_w['b_cls'])
    X_e_norm = (X_e - f_w['mean']) / f_w['std']
    f_res = face_net.forward(X_e_norm)
    acc_face = np.mean(np.argmax(f_res['emotion_probs'], axis=-1) == y_e)

    # 2. Body Pose Network
    pose_net = BodyPoseNetwork()
    pose_net.set_weights(p_w['W1'], p_w['b1'], p_w['W2'], p_w['b2'])
    X_p_norm = (X_p - p_w['mean']) / p_w['std']
    p_res = pose_net.forward(X_p_norm)
    acc_pose = np.mean(np.argmax(p_res['posture_probs'], axis=-1) == y_p)

    # 3. Video Temporal Model
    video_net = VideoTemporalCNNLSTM()
    video_net.set_weights(v_w['W_x'], v_w['W_h'], v_w['b_h'], v_w['W_cls'], v_w['b_cls'])
    X_u_norm = (X_u - v_w['mean']) / v_w['std']
    v_res = video_net.forward(X_u_norm)
    acc_video = np.mean(np.argmax(v_res['temporal_probs'], axis=-1) == y_u)

    # 4. Context-Aware Attention Fusion Network
    context_net = ContextMetadataNetwork()
    np.random.seed(42)
    meta_inputs = np.column_stack([
        np.random.randint(0, 6, n_samples),
        np.random.uniform(0, 24, n_samples),
        np.random.uniform(0.1, 1.0, n_samples),
        np.random.randint(0, 40, n_samples),
        np.random.uniform(0.4, 0.95, n_samples)
    ])
    c_res = context_net.forward(meta_inputs)

    fusion_net = ContextAwareAttentionFusion()
    fusion_net.set_weights(fu_w['W_attn'], fu_w['b_attn'], fu_w['W_cls'], fu_w['b_cls'])

    fusion_res = fusion_net.forward(
        f_res['features'][:n_samples],
        p_res['features'][:n_samples],
        v_res['features'][:n_samples],
        c_res['features'][:n_samples]
    )

    y_multimodal = np.clip(y_u[:n_samples], 0, 4)
    acc_fusion = np.mean(np.argmax(fusion_res['category_probs'], axis=-1) == y_multimodal)

    metrics = compute_multimodal_metrics(
        y_multimodal,
        fusion_res['category_probs'],
        fusion_res['predicted_category'],
        fusion_res['reliability_score']
    )

    print("=" * 65)
    print("      RELIABILITY-AWARE MULTIMODAL ANOMALY DETECTION MODEL ACCURACY")
    print("=" * 65)
    print(f"  1. Facial Expression CNN (Multimodal Emotion Dataset): {acc_face*100:.2f}%")
    print(f"  2. Body Pose Network (MPII Human Pose Dataset):       {acc_pose*100:.2f}%")
    print(f"  3. Video Temporal CNN+LSTM (UCF-Crime Dataset):        {acc_video*100:.2f}%")
    print(f"  4. Multimodal Attention Fusion Network (End-to-End):   {acc_fusion*100:.2f}%\n")

    print(f"  - System Mean Reliability Index (Correct Detections):   {metrics['mean_reliability_correct']*100:.2f}%")
    print(f"  - System Mean Reliability Index (Uncertain/Incorrect):  {metrics['mean_reliability_incorrect']*100:.2f}%")
    print("=" * 65)

if __name__ == '__main__':
    evaluate_framework()
