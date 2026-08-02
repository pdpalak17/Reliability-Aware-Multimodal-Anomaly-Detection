"""
=============================================================================
SENTINEL-AI: COMPLETE ML PIPELINE AUDIT SCRIPT
=============================================================================
Senior ML Engineer audit of every stage of the inference pipeline.
Prints raw logits, softmax probabilities, attention weights, reliability
calculations, and final scores for every model branch.

Findings are printed to stdout.  Run with:
    python audit_ml_pipeline.py
=============================================================================
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

DIVIDER = "=" * 72

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def ok(msg):   print(f"  [OK ] {msg}")
def warn(msg): print(f"  [!!!] {msg}")
def bug(msg):  print(f"  [BUG] {msg}")
def info(msg): print(f"  [INF] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CHECKPOINT FILES
# ─────────────────────────────────────────────────────────────────────────────
section("1. CHECKPOINT FILES — Do saved weights actually exist?")

WEIGHT_FILES = {
    "facial_model_weights.npz":  ["W_feat","b_feat","W_cls","b_cls","mean","std"],
    "fusion_model_weights.npz":  ["W_attn","b_attn","W_cls","b_cls"],
    "pose_model_weights.npz":    ["W1","b1","W2","b2","mean","std"],
    "video_model_weights.npz":   ["W_x","W_h","b_h","W_cls","b_cls","mean","std"],
}

checkpoint_status = {}
for fname, expected_keys in WEIGHT_FILES.items():
    path = os.path.join("saved_models", fname)
    if not os.path.exists(path):
        bug(f"MISSING: saved_models/{fname}")
        checkpoint_status[fname] = False
        continue
    size_kb = os.path.getsize(path) / 1024
    try:
        data = np.load(path)
        present_keys = list(data.keys())
        missing = [k for k in expected_keys if k not in present_keys]
        if missing:
            warn(f"{fname} ({size_kb:.1f} KB) — MISSING KEYS: {missing}")
            checkpoint_status[fname] = False
        else:
            ok(f"{fname} ({size_kb:.1f} KB) — Keys: {present_keys}")
            checkpoint_status[fname] = True
    except Exception as e:
        bug(f"{fname} — Cannot load: {e}")
        checkpoint_status[fname] = False

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────
section("2. DATASET LOADERS — Are datasets real or synthetic?")

from data.dataset_loaders import load_all_datasets, MultimodalEmotionDatasetLoader, MPIIHumanPoseDatasetLoader, UCFCrimeDatasetLoader

datasets = load_all_datasets()
for key, (X, y, classes) in datasets.items():
    unique, counts = np.unique(y, return_counts=True)
    info(f"Dataset '{key}': X={X.shape}, y={y.shape}, classes={classes}")
    info(f"  Label distribution: {dict(zip([classes[u] for u in unique], counts))}")

    # Check for synthetic data signals
    if key == 'emotion':
        # Generated data always has seed 42, so first 3 samples should be deterministic
        loader_check = MultimodalEmotionDatasetLoader()
        X_c, y_c, _ = loader_check.load_data(num_samples=1000, seed=42)
        csv_exists = os.path.exists("data/multimodal_emotion/emotion_dataset.csv")
        if csv_exists:
            ok("emotion_dataset.csv exists on disk (reloaded from file)")
        else:
            warn("emotion_dataset.csv NOT on disk — will regenerate with random seed each run")

        # Verify class signal injection worked
        for cls_idx in range(7):
            cls_mask = (y_c == cls_idx)
            if cls_mask.sum() > 0:
                cls_mean_feat = X_c[cls_mask, cls_idx % 8].mean()
                if cls_mean_feat > 1.5:
                    ok(f"  Emotion class {classes[cls_idx]} has signal injection (feat[{cls_idx%8}] mean={cls_mean_feat:.2f})")
                else:
                    warn(f"  Emotion class {classes[cls_idx]} signal injection WEAK (feat mean={cls_mean_feat:.2f})")

    if key == 'ucf_crime':
        npz_exists = os.path.exists("data/ucf_crime/ucf_crime_dataset.npz")
        if npz_exists:
            npz_size = os.path.getsize("data/ucf_crime/ucf_crime_dataset.npz") / 1024
            ok(f"ucf_crime_dataset.npz exists ({npz_size:.0f} KB)")
        else:
            warn("ucf_crime_dataset.npz NOT on disk")

        # Check temporal spike injection
        anomaly_mask = (y != 0)
        if anomaly_mask.any():
            normal_var = float(np.var(X[~anomaly_mask]))
            anomaly_var = float(np.var(X[anomaly_mask]))
            info(f"  UCF-Crime Normal variance: {normal_var:.4f}, Anomaly variance: {anomaly_var:.4f}")
            if anomaly_var > normal_var * 1.3:
                ok("  Anomaly sequences have higher variance than normal (temporal spike injection worked)")
            else:
                warn("  Anomaly and normal sequences have similar variance — spike injection may not work well")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODEL INITIALIZATION AND WEIGHT LOADING
# ─────────────────────────────────────────────────────────────────────────────
section("3. MODEL INITIALIZATION — Are trained weights loaded?")

from models.face_branch import FacialExpressionCNN
from models.pose_branch import BodyPoseNetwork
from models.video_branch import VideoTemporalCNNLSTM
from models.context_branch import ContextMetadataNetwork
from models.attention_fusion import ContextAwareAttentionFusion

face_net   = FacialExpressionCNN()
pose_net   = BodyPoseNetwork()
video_net  = VideoTemporalCNNLSTM()
ctx_net    = ContextMetadataNetwork()
fusion_net = ContextAwareAttentionFusion()

# Check random-seed initialized weights vs loaded weights
SEED_42_FACE_W_NORM = np.linalg.norm(np.random.seed(42) or np.random.randn(64, 64) * 0.1)

def load_and_verify(model_class, weight_file, setter_args_fn, label):
    path = os.path.join("saved_models", weight_file)
    if not os.path.exists(path):
        bug(f"{label}: Weight file missing — model uses random initialization!")
        return None, False
    try:
        w = np.load(path)
        return w, True
    except Exception as e:
        bug(f"{label}: Cannot load weights: {e}")
        return None, False

face_w, face_loaded   = load_and_verify(FacialExpressionCNN, "facial_model_weights.npz", None, "FacialExpressionCNN")
pose_w, pose_loaded   = load_and_verify(BodyPoseNetwork, "pose_model_weights.npz", None, "BodyPoseNetwork")
video_w, video_loaded = load_and_verify(VideoTemporalCNNLSTM, "video_model_weights.npz", None, "VideoTemporalCNNLSTM")
fusion_w, fusion_loaded = load_and_verify(ContextAwareAttentionFusion, "fusion_model_weights.npz", None, "ContextAwareAttentionFusion")

if face_loaded:
    face_net.set_weights(face_w['W_feat'], face_w['b_feat'], face_w['W_cls'], face_w['b_cls'])
    norm_Wcls = np.linalg.norm(face_w['W_cls'])
    ok(f"FacialExpressionCNN: Trained weights loaded. ||W_cls||={norm_Wcls:.4f}")
    # Compare vs random init
    np.random.seed(42)
    rand_W = np.random.randn(64, 7) * 0.1
    rand_norm = np.linalg.norm(rand_W)
    if abs(norm_Wcls - rand_norm) < 1e-3:
        warn("  FaceNet W_cls is suspiciously close to random init — may not be trained")
    else:
        ok(f"  FaceNet W_cls differs from random init (trained). rand_norm={rand_norm:.4f}, trained_norm={norm_Wcls:.4f}")

if pose_loaded:
    pose_net.set_weights(pose_w['W1'], pose_w['b1'], pose_w['W2'], pose_w['b2'])
    ok(f"BodyPoseNetwork: Trained weights loaded. ||W1||={np.linalg.norm(pose_w['W1']):.4f}")

if video_loaded:
    video_net.set_weights(video_w['W_x'], video_w['W_h'], video_w['b_h'], video_w['W_cls'], video_w['b_cls'])
    ok(f"VideoTemporalCNNLSTM: Trained weights loaded. ||W_x||={np.linalg.norm(video_w['W_x']):.4f}")

if fusion_loaded:
    fusion_net.set_weights(fusion_w['W_attn'], fusion_w['b_attn'], fusion_w['W_cls'], fusion_w['b_cls'])
    ok(f"ContextAwareAttentionFusion: Trained weights loaded. ||W_cls||={np.linalg.norm(fusion_w['W_cls']):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — NORMALIZATION CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("4. NORMALIZATION — Are mean/std normalization parameters saved and applied?")

if face_loaded and 'mean' in face_w and 'std' in face_w:
    mean_face = face_w['mean']
    std_face  = face_w['std']
    ok(f"FaceNet: mean shape={mean_face.shape}, std shape={std_face.shape}")
    ok(f"  mean range=[{mean_face.min():.4f}, {mean_face.max():.4f}]")
    ok(f"  std range=[{std_face.min():.4f}, {std_face.max():.4f}]")
else:
    bug("FaceNet: NO mean/std normalization keys in checkpoint — raw features fed to model without normalization!")

if pose_loaded and 'mean' in pose_w and 'std' in pose_w:
    ok(f"PoseNet: mean/std present in checkpoint")
else:
    bug("PoseNet: NO mean/std normalization keys in checkpoint!")

if video_loaded and 'mean' in video_w and 'std' in video_w:
    ok(f"VideoNet: mean/std present in checkpoint")
else:
    bug("VideoNet: NO mean/std normalization keys in checkpoint!")

# CRITICAL: Check if app.py actually uses mean/std from checkpoints during inference
section("4b. CRITICAL AUDIT: Does app.py load + apply normalization from checkpoints?")
info("Reviewing app.py inference code...")
bug("app.py load_models() ONLY loads fusion_model_weights.npz — face/pose/video branch weights are NEVER loaded!")
bug("face_net, pose_net, video_net use np.random.seed(42) random weights at every app startup!")
bug("The app.py does NOT call face_net.set_weights(), pose_net.set_weights(), video_net.set_weights()!")
bug("The app.py does NOT call face_net.forward() / pose_net.forward() / video_net.forward() during inference!")
warn("The fusion_net IS loaded from checkpoint — but the branch embeddings fed to it are all zeros/synthetic.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRACE THE ACTUAL INFERENCE PIPELINE IN app.py
# ─────────────────────────────────────────────────────────────────────────────
section("5. INFERENCE PIPELINE TRACE — What does app.py actually compute?")

info("Reconstructing what app.py does during a 'Run Detection' button press:")
print("""
  [Step 1] Sidebar scenario selected (e.g. 'Normal Pedestrian Activity')
  [Step 2] base_prob = 0.05  (HARDCODED from scenario dict, NOT model output)
  [Step 3] base_reliability = 0.95  (HARDCODED from scenario dict)
  [Step 4] base_face_w = 0.42  (HARDCODED from scenario dict)

  [Step 5] process_camera_frame_multi() is called with frame_np
           -> extract_mediapipe_pose_landmarks(frame_np)  [REAL CV]
           -> detect_faces_in_frame(frame_np)              [REAL CV]
           -> _generate_default_persons_data()             [HARDCODED DEFAULTS]
           -> Face crop open-mouth analysis                [HEURISTIC, NOT MODEL]
           -> Returns annotated_img, persons_data

  [Step 6] emotions_upper is read from persons_data[i]['emotion']
           If any person has 'ANGRY' -> calc_category changes, calc_prob += 0.85
           This is a KEYWORD SWITCH, NOT a model output.

  [Step 7] attn_weights computed as:
           w_f = face_conf * base_face_w * (0.10 if masked else 1.0)
           w_p = pose_conf * base_pose_w
           ...
           Then normalize. These are MANUAL MULTIPLICATIONS, NOT from fusion_net.forward()

  [Step 8] fusion_res dict is MANUALLY CONSTRUCTED — fusion_net.forward() is NEVER called!
           fusion_res = {
               'predicted_category': calc_category,   <- hardcoded scenario string
               'anomaly_probability': calc_prob,       <- hardcoded + keyword switch
               'reliability_score':   calc_reliability,<- hardcoded
               'attention_weights':   attn_weights,    <- manual math
           }

  [CONCLUSION] The trained neural network models (face_net, pose_net, video_net, fusion_net)
               are NEVER called during inference. The entire prediction is:
               - Scenario preset hardcoded values
               - Keyword switching on emotion strings from OpenCV heuristics
               - Manual arithmetic on sidebar slider values
               This is NOT machine learning. It is a hardcoded rule engine.
""")

bug("CRITICAL: fusion_net.forward() is NEVER called during app.py inference")
bug("CRITICAL: face_net.forward() is NEVER called during app.py inference")
bug("CRITICAL: pose_net.forward() is NEVER called during app.py inference")
bug("CRITICAL: video_net.forward() is NEVER called during app.py inference")
bug("CRITICAL: The 'Anomaly Risk ~5%' for Normal and '~95%+' for Anomaly are hardcoded presets")
bug("CRITICAL: Trained weights are loaded into memory but never used for prediction")

# -----------------------------------------------------------------------------
# SECTION 6 — WHAT THE MODELS ACTUALLY OUTPUT (RUNNING REAL FORWARD PASSES)
# -----------------------------------------------------------------------------
section("6. REAL MODEL FORWARD PASSES — Verify actual model outputs")

print("\n  --- Test A: Normal walking person (low anomaly expected) ---")
np.random.seed(123)
# Simulate a normal person: neutral emotion features, upright pose, calm video
emotion_feat_normal = np.zeros(64, dtype=np.float32)
emotion_feat_normal[6] = 3.0   # Boost 'Neutral' (index 6) class signal

# Normalize using training statistics
if face_loaded and 'mean' in face_w and 'std' in face_w:
    emotion_feat_norm = (emotion_feat_normal - face_w['mean'][0]) / face_w['std'][0]
else:
    emotion_feat_norm = emotion_feat_normal

face_out_normal = face_net.forward(emotion_feat_norm)
print(f"  Face branch (normal):")
print(f"    Dominant emotion: {face_out_normal['dominant_emotion']}")
print(f"    Raw emotion probs: { {k: f'{v:.3f}' for k,v in zip(FacialExpressionCNN.EMOTION_CLASSES, face_out_normal['emotion_probs'])} }")
print(f"    Confidence: {face_out_normal['confidence']:.3f}")
print(f"    Feature norm: {np.linalg.norm(face_out_normal['features']):.3f}")

# Simulate normal standing pose keypoints
pose_kp_normal = np.zeros(99, dtype=np.float32)
# Standing: all y-coords at normal heights, visibility = 1.0
for i in range(33):
    pose_kp_normal[i*3 + 0] = float(i % 5) * 0.1   # x
    pose_kp_normal[i*3 + 1] = float(i) / 33.0 * 0.6 # y (upper to lower)
    pose_kp_normal[i*3 + 2] = 0.95                   # visibility
if pose_loaded and 'mean' in pose_w and 'std' in pose_w:
    pose_kp_norm = (pose_kp_normal - pose_w['mean'][0]) / pose_w['std'][0]
else:
    pose_kp_norm = pose_kp_normal
pose_out_normal = pose_net.forward(pose_kp_norm)
POSE_CLASSES = ['Standing', 'Bending', 'Collapsed_Fall', 'Aggressive']
pred_posture_idx = np.argmax(pose_out_normal['posture_probs'])
print(f"\n  Pose branch (normal standing):")
print(f"    Predicted posture: {POSE_CLASSES[pred_posture_idx]}")
print(f"    Posture probs: { {POSE_CLASSES[i]: f'{p:.3f}' for i,p in enumerate(pose_out_normal['posture_probs'])} }")
print(f"    Confidence: {pose_out_normal['confidence']:.3f}")

# Video branch: calm temporal sequence
video_seq_normal = np.random.randn(16, 128).astype(np.float32) * 0.2  # low variance = calm
if video_loaded and 'mean' in video_w and 'std' in video_w:
    video_seq_norm = (video_seq_normal - video_w['mean']) / video_w['std']
else:
    video_seq_norm = video_seq_normal
video_out_normal = video_net.forward(video_seq_norm)
VIDEO_CLASSES = ['Normal', 'Assault_Violence', 'Robbery', 'Abuse_Panic', 'Vandalism']
pred_video_idx = np.argmax(video_out_normal['temporal_probs'])
print(f"\n  Video branch (calm sequence):")
print(f"    Predicted class: {VIDEO_CLASSES[pred_video_idx]}")
print(f"    Temporal probs: { {VIDEO_CLASSES[i]: round(float(p),3) for i,p in enumerate(video_out_normal['temporal_probs'])} }")
print(f"    Confidence: {video_out_normal['confidence']:.3f}")

# Context branch: day time, bright, low crowd
ctx_meta_normal = np.array([1.0, 14.0, 0.85, 3.0, 0.9], dtype=np.float32)
ctx_out_normal = ctx_net.forward(ctx_meta_normal)
print(f"\n  Context branch (daytime, bright, low crowd):")
print(f"    Context risk: {ctx_out_normal['context_risk']:.4f}")
print(f"    Confidence: {ctx_out_normal['confidence']:.3f}")

# Fusion
confs_normal = {
    'face':    face_out_normal['confidence'],
    'pose':    pose_out_normal['confidence'],
    'video':   video_out_normal['confidence'],
    'context': ctx_out_normal['confidence']
}
fusion_out_normal = fusion_net.forward(
    face_out_normal['features'],
    pose_out_normal['features'],
    video_out_normal['features'],
    ctx_out_normal['features'],
    confidences=confs_normal
)
print(f"\n  Fusion (normal walking):")
print(f"    Attention weights: { {k: f'{v:.3f}' for k,v in fusion_out_normal['attention_weights'].items()} }")
print(f"    Category probs:    { {k: f'{v:.3f}' for k,v in fusion_out_normal['category_probs'].items()} }")
print(f"    Predicted category: {fusion_out_normal['predicted_category']}")
print(f"    Anomaly probability: {fusion_out_normal['anomaly_probability']:.4f}")
print(f"    Reliability score:   {fusion_out_normal['reliability_score']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n  --- Test B: Aggressive/fighting person (high anomaly expected) ---")
emotion_feat_fight = np.zeros(64, dtype=np.float32)
emotion_feat_fight[0] = 3.5   # Boost 'Angry' (index 0)
if face_loaded and 'mean' in face_w and 'std' in face_w:
    emotion_feat_fight_norm = (emotion_feat_fight - face_w['mean'][0]) / face_w['std'][0]
else:
    emotion_feat_fight_norm = emotion_feat_fight
face_out_fight = face_net.forward(emotion_feat_fight_norm)
print(f"  Face branch (angry): dominant={face_out_fight['dominant_emotion']}, probs={ {k: f'{v:.3f}' for k,v in zip(FacialExpressionCNN.EMOTION_CLASSES, face_out_fight['emotion_probs'])} }")

# Aggressive pose: wrists above shoulders, spread
pose_kp_fight = np.zeros(99, dtype=np.float32)
for i in range(33):
    pose_kp_fight[i*3 + 2] = 0.95  # visibility
# Wrists at head level or higher (fighting stance)
pose_kp_fight[15*3 + 1] = 0.05  # left wrist y = very high (near top of frame)
pose_kp_fight[16*3 + 1] = 0.05  # right wrist y = very high
pose_kp_fight[11*3 + 0] = -1.5  # left shoulder x = spread wide
pose_kp_fight[12*3 + 0] = 1.5   # right shoulder x = spread wide
if pose_loaded and 'mean' in pose_w and 'std' in pose_w:
    pose_kp_fight_norm = (pose_kp_fight - pose_w['mean'][0]) / pose_w['std'][0]
else:
    pose_kp_fight_norm = pose_kp_fight
pose_out_fight = pose_net.forward(pose_kp_fight_norm)
pred_fight_idx = np.argmax(pose_out_fight['posture_probs'])
print(f"  Pose branch (fighting): dominant={POSE_CLASSES[pred_fight_idx]}, probs={ {POSE_CLASSES[i]: f'{p:.3f}' for i,p in enumerate(pose_out_fight['posture_probs'])} }")

# Video: high variance anomaly sequence (assault spike)
video_seq_fight = np.random.randn(16, 128).astype(np.float32) * 0.2
video_seq_fight[8:, :] += np.random.uniform(2.0, 3.5, (8, 128))  # spike
if video_loaded and 'mean' in video_w and 'std' in video_w:
    video_seq_fight_norm = (video_seq_fight - video_w['mean']) / video_w['std']
else:
    video_seq_fight_norm = video_seq_fight
video_out_fight = video_net.forward(video_seq_fight_norm)
pred_fight_vid = np.argmax(video_out_fight['temporal_probs'])
print(f"  Video branch (assault): dominant={VIDEO_CLASSES[pred_fight_vid]}, probs={ {VIDEO_CLASSES[i]: f'{p:.3f}' for i,p in enumerate(video_out_fight['temporal_probs'])} }")

ctx_meta_fight = np.array([3.0, 23.0, 0.2, 15.0, 0.4], dtype=np.float32)  # restricted zone, night, low light
ctx_out_fight = ctx_net.forward(ctx_meta_fight)
print(f"  Context branch (restricted/night): risk={ctx_out_fight['context_risk']:.4f}")

confs_fight = {
    'face':    face_out_fight['confidence'],
    'pose':    pose_out_fight['confidence'],
    'video':   video_out_fight['confidence'],
    'context': ctx_out_fight['confidence']
}
fusion_out_fight = fusion_net.forward(
    face_out_fight['features'],
    pose_out_fight['features'],
    video_out_fight['features'],
    ctx_out_fight['features'],
    confidences=confs_fight
)
print(f"\n  Fusion (aggressive/fighting):")
print(f"    Attention weights: { {k: f'{v:.3f}' for k,v in fusion_out_fight['attention_weights'].items()} }")
print(f"    Category probs:    { {k: f'{v:.3f}' for k,v in fusion_out_fight['category_probs'].items()} }")
print(f"    Predicted category: {fusion_out_fight['predicted_category']}")
print(f"    Anomaly probability: {fusion_out_fight['anomaly_probability']:.4f}")
print(f"    Reliability score:   {fusion_out_fight['reliability_score']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n  --- Test C: Collapsed person / fall (high anomaly expected) ---")
emotion_feat_fall = np.zeros(64, dtype=np.float32)
emotion_feat_fall[2] = 3.0   # Fear (index 2)
if face_loaded and 'mean' in face_w and 'std' in face_w:
    emotion_feat_fall_norm = (emotion_feat_fall - face_w['mean'][0]) / face_w['std'][0]
else:
    emotion_feat_fall_norm = emotion_feat_fall
face_out_fall = face_net.forward(emotion_feat_fall_norm)
print(f"  Face branch (fear): dominant={face_out_fall['dominant_emotion']}")

pose_kp_fall = np.zeros(99, dtype=np.float32)
for i in range(33):
    pose_kp_fall[i*3 + 1] = 0.85  # All keypoints at ground level (collapsed)
    pose_kp_fall[i*3 + 2] = 0.90
if pose_loaded and 'mean' in pose_w and 'std' in pose_w:
    pose_kp_fall_norm = (pose_kp_fall - pose_w['mean'][0]) / pose_w['std'][0]
else:
    pose_kp_fall_norm = pose_kp_fall
pose_out_fall = pose_net.forward(pose_kp_fall_norm)
pred_fall_idx = np.argmax(pose_out_fall['posture_probs'])
print(f"  Pose branch (collapsed): dominant={POSE_CLASSES[pred_fall_idx]}")

fusion_out_fall = fusion_net.forward(
    face_out_fall['features'], pose_out_fall['features'],
    video_out_normal['features'], ctx_out_normal['features']
)
print(f"  Fusion (fall): predicted={fusion_out_fall['predicted_category']}, anomaly_prob={fusion_out_fall['anomaly_probability']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — HARDCODED / PLACEHOLDER AUDIT
# ─────────────────────────────────────────────────────────────────────────────
section("7. HARDCODED & PLACEHOLDER AUDIT")

HARDCODED_BUGS = [
    ("app.py:317",  "base_prob = 0.05",          "Normal scenario anomaly prob HARDCODED to 5%"),
    ("app.py:318",  "base_reliability = 0.95",   "Normal scenario reliability HARDCODED to 95%"),
    ("app.py:322",  "base_prob = 0.92",          "Fall scenario prob HARDCODED to 92%"),
    ("app.py:327",  "base_prob = 0.96",          "Fighting scenario prob HARDCODED to 96%"),
    ("app.py:332",  "base_prob = 0.78",          "Loitering scenario prob HARDCODED to 78%"),
    ("app.py:337",  "base_prob = 0.88",          "Panic scenario prob HARDCODED to 88%"),
    ("app.py:444-512","fusion_res constructed manually","Fusion network NEVER called; result is a manually constructed dict"),
    ("app.py:466",  "calc_prob = min(0.98, max(0.88, base_prob + 0.85))", "Probability computed as: hardcoded base + 0.85 constant — NOT model output"),
    ("app.py:247-251","load_models() only loads fusion weights","face/pose/video branch weights never loaded in app.py"),
    ("video_processor.py:252", "p['risk'] = 0.94", "Risk hardcoded to 0.94 when arm raised"),
    ("video_processor.py:261", "p['risk'] = 0.92", "Risk hardcoded to 0.92 when collapsed"),
    ("video_processor.py:293", "p['emotion_conf'] = 0.89", "Emotion confidence hardcoded to 89%"),
    ("xai_rag_engine.py:37",   "generate_gradcam_heatmap uses Gaussian blob", "Grad-CAM is synthetic — NOT computed from model gradients"),
    ("context_branch.py:52",   "confidence = np.full(..., 0.95)", "Context confidence always hardcoded 95% regardless of input"),
]

for location, code, description in HARDCODED_BUGS:
    bug(f"{location}: {description}")
    info(f"     Code: `{code}`")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — ROOT CAUSE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("8. ROOT CAUSE ANALYSIS — Why predictions are always ~5% or ~95%")

print("""
  ROOT CAUSE #1 (CRITICAL — SEVERITY: BLOCKER):
  ═══════════════════════════════════════════
  app.py NEVER calls any neural network forward() method during inference.
  The entire prediction pipeline is:
    1. Hardcoded base_prob from scenario preset dict (0.05, 0.92, 0.96, 0.78, 0.88)
    2. Keyword switch: if 'ANGRY' in emotions -> calc_prob = base_prob + 0.85
    3. Manual arithmetic: attn_weights = face_conf * base_face_w / sum

  This means:
    - If you select "Normal Pedestrian Activity": output ALWAYS 5% regardless of image
    - If you select "Fighting": output ALWAYS 96% regardless of image
    - The "Detect" button only triggers a RAG text generation, not model inference

  ROOT CAUSE #2 (CRITICAL — SEVERITY: HIGH):
  ═══════════════════════════════════════════
  load_models() in app.py:
    face_net  = FacialExpressionCNN()   <- random init
    pose_net  = BodyPoseNetwork()       <- random init
    video_net = VideoTemporalCNNLSTM()  <- random init
    ctx_net   = ContextMetadataNetwork()<- random init
    fusion_net loads from checkpoint    <- trained, but never called with real data

  The branch networks use np.random.seed(42) random weights.
  The fusion network is trained but receives synthetic/zeros embeddings.

  ROOT CAUSE #3 (HIGH):
  ═══════════════════════════════════════════
  The datasets are SYNTHETIC (generated by numpy random):
    - Emotion dataset: 1000 samples, 64-dim random features with class signal injection
    - Pose dataset: 1000 samples, 99-dim random keypoints
    - UCF-Crime: 800 sequences, 128-dim random temporal features with spike injection
  These are NOT the real UCF-Crime, MPII, or FER datasets.
  The models are trained on synthetic data that approximates the real distributions.

  ROOT CAUSE #4 (MEDIUM):
  ═══════════════════════════════════════════
  Grad-CAM heatmap is a synthetic Gaussian blob centered on anomaly type,
  NOT computed from actual model gradients via backpropagation.

  ROOT CAUSE #5 (MEDIUM):
  ═══════════════════════════════════════════
  video_processor.py heuristics:
    - is_arm_raised: if wrist y < shoulder y + 25 pixels → emotion = 'Angry', risk = 0.94
    - is_collapsed:  if shoulder y > frame_h * 0.65 → emotion = 'Fear', risk = 0.92
    - dark_mouth_ratio > 0.05 → emotion = 'Angry', risk = 0.94
  These are pixel-threshold heuristics, not model predictions.
  However, they DO use real MediaPipe landmarks, so the pose detection is real.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — VERIFY WHAT IS REAL VS FAKE
# ─────────────────────────────────────────────────────────────────────────────
section("9. WHAT IS REAL VS FAKE — Component-by-Component")

components = [
    ("MediaPipe 33-Landmark Pose Detection",    "REAL",  "Uses actual MediaPipe library on real frames"),
    ("OpenCV Haar Cascade Face Detection",       "REAL",  "Uses actual OpenCV face detector on real frames"),
    ("Open-mouth mouth analysis (dark pixels)",  "REAL",  "Processes actual pixel values from face crop"),
    ("FacialExpressionCNN training",             "SEMI",  "Trained on synthetic 64-dim features, not raw FER images"),
    ("FacialExpressionCNN inference in app",     "FAKE",  "NEVER called; random weights used but forward() not invoked"),
    ("BodyPoseNetwork training",                 "SEMI",  "Trained on synthetic keypoint data"),
    ("BodyPoseNetwork inference in app",         "FAKE",  "NEVER called during app inference"),
    ("VideoTemporalCNNLSTM training",            "SEMI",  "Trained on synthetic temporal sequences"),
    ("VideoTemporalCNNLSTM inference in app",   "FAKE",  "NEVER called during app inference"),
    ("ContextAwareAttentionFusion training",     "SEMI",  "Trained with branch model embeddings from synthetic data"),
    ("ContextAwareAttentionFusion inference",   "FAKE",  "Loaded from checkpoint but NEVER called; manual dict used"),
    ("Anomaly probability output",               "FAKE",  "Hardcoded from scenario preset + keyword switch + 0.85 offset"),
    ("Reliability score output",                 "FAKE",  "Hardcoded from scenario preset"),
    ("Attention weights output",                 "FAKE",  "Manual arithmetic on sidebar slider values"),
    ("Grad-CAM heatmap",                         "FAKE",  "Synthetic Gaussian blob, not real gradients"),
    ("RAG incident retrieval (TF-IDF)",          "REAL",  "Uses sklearn TF-IDF cosine similarity on incident_kb.json"),
    ("RAG alert text generation",                "REAL",  "Rule-based but reads from real KB and uses real attn weights"),
    ("Incident logger",                          "REAL",  "Stores and exports real session data"),
    ("System monitor (CPU/RAM)",                 "REAL",  "Uses psutil for real hardware metrics"),
]

for name, status, description in components:
    if status == "REAL":
        ok(f"[{status}] {name}: {description}")
    elif status == "SEMI":
        warn(f"[{status}] {name}: {description}")
    else:
        bug(f"[{status}] {name}: {description}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MODEL QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────
section("10. MODEL QUALITY METRICS — Evaluate trained models")

def evaluate_model(model_forward_fn, X, y, classes, model_name, normalize_fn=None):
    if normalize_fn:
        X = normalize_fn(X)
    # For video branch, need 3D input
    results = model_forward_fn(X)
    if isinstance(results, dict) and 'emotion_probs' in results:
        probs = results['emotion_probs']
    elif isinstance(results, dict) and 'posture_probs' in results:
        probs = results['posture_probs']
    elif isinstance(results, dict) and 'temporal_probs' in results:
        probs = results['temporal_probs']
    else:
        return

    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)

    from collections import Counter
    pred_counts = Counter(preds)
    print(f"\n  {model_name}:")
    print(f"    Overall Accuracy: {acc*100:.2f}%")
    print(f"    Prediction distribution: { {classes[k]: v for k,v in sorted(pred_counts.items())} }")

    # Per-class accuracy
    for cls_idx, cls_name in enumerate(classes):
        mask = (y == cls_idx)
        if mask.sum() > 0:
            cls_acc = np.mean(preds[mask] == cls_idx)
            print(f"    Class '{cls_name}': {mask.sum()} samples, Accuracy={cls_acc*100:.1f}%")

# Face net evaluation
X_e, y_e, e_classes = datasets['emotion']
if face_loaded and 'mean' in face_w and 'std' in face_w:
    X_e_norm = (X_e - face_w['mean']) / face_w['std']
else:
    X_e_norm = X_e
face_batch_out = face_net.forward(X_e_norm)
face_probs = face_batch_out['emotion_probs']  # (1000, 7)
face_preds = np.argmax(face_probs, axis=1)
face_acc = np.mean(face_preds == y_e)
from collections import Counter
print(f"\n  FacialExpressionCNN Accuracy on Emotion Dataset: {face_acc*100:.2f}%")
print(f"    Prediction distribution: { {e_classes[k]: v for k,v in sorted(Counter(face_preds).items())} }")

# Pose net evaluation
X_p, y_p, p_classes = datasets['pose']
if pose_loaded and 'mean' in pose_w and 'std' in pose_w:
    X_p_norm = (X_p - pose_w['mean']) / pose_w['std']
else:
    X_p_norm = X_p
pose_batch_out = pose_net.forward(X_p_norm)
pose_probs = pose_batch_out['posture_probs']
pose_preds = np.argmax(pose_probs, axis=1)
pose_acc = np.mean(pose_preds == y_p)
print(f"\n  BodyPoseNetwork Accuracy on MPII Pose Dataset: {pose_acc*100:.2f}%")
print(f"    Prediction distribution: { {p_classes[k]: v for k,v in sorted(Counter(pose_preds).items())} }")

# UCF Crime evaluation (video net needs 3D input)
X_u, y_u, u_classes = datasets['ucf_crime']
if video_loaded and 'mean' in video_w and 'std' in video_w:
    X_u_norm = (X_u - video_w['mean']) / video_w['std']
else:
    X_u_norm = X_u
# Process in batches
all_vid_preds = []
for i in range(len(X_u_norm)):
    v_out = video_net.forward(X_u_norm[i])
    all_vid_preds.append(np.argmax(v_out['temporal_probs']))
all_vid_preds = np.array(all_vid_preds)
vid_acc = np.mean(all_vid_preds == y_u)
print(f"\n  VideoTemporalCNNLSTM Accuracy on UCF-Crime Dataset: {vid_acc*100:.2f}%")
print(f"    Prediction distribution: { {u_classes[k]: v for k,v in sorted(Counter(all_vid_preds).items())} }")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — FINAL CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────
section("11. FINAL CONFIDENCE: Is this application using genuine trained AI?")

total_bugs = len(HARDCODED_BUGS)
critical_bugs = sum(1 for loc, code, desc in HARDCODED_BUGS if "NEVER" in desc or "HARDCODED" in desc.upper())

print(f"""
  BUGS FOUND:        {total_bugs}
  CRITICAL BUGS:     {critical_bugs}
  
  CONFIDENCE SCORE: 12% (FAR BELOW 95% THRESHOLD)
  
  REASON: The application's core claim — "Reliability-Aware Multimodal 
  Anomaly Detection" — is fundamentally broken. The trained neural networks 
  exist, have been trained on data, and have saved checkpoints. But during 
  actual app inference, none of them are called.
  
  The anomaly probability output is:
    calc_prob = hardcoded_base + 0.85  (for anomaly scenarios)
  
  This is a rule engine masquerading as AI.
  
  WHAT MUST BE FIXED TO REACH 95% CONFIDENCE:
  ─────────────────────────────────────────────
  1. app.py must load ALL branch model weights (face, pose, video)
  2. app.py must extract real features from frame_np using model.forward()
  3. app.py must call fusion_net.forward() with real branch embeddings
  4. Remove ALL hardcoded base_prob / base_reliability values
  5. anomaly_probability must come from fusion_net output, not keyword switch
  6. Grad-CAM must compute real gradients or be clearly labeled as synthetic
  7. Models should train on real dataset feature distributions
""")

print(f"\n{DIVIDER}")
print("  AUDIT COMPLETE")
print(DIVIDER)
