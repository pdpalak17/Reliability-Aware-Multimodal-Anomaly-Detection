import numpy as np
import sys, os
sys.path.insert(0, 'e:/Multimodel')
os.chdir('e:/Multimodel')

from models.attention_fusion import ContextAwareAttentionFusion
from models.face_branch import FacialExpressionCNN
from models.pose_branch import BodyPoseNetwork
from models.video_branch import VideoTemporalCNNLSTM
from models.context_branch import ContextMetadataNetwork
from data.dataset_loaders import load_all_datasets

fw  = np.load('saved_models/facial_model_weights.npz')
pw  = np.load('saved_models/pose_model_weights.npz')
vw  = np.load('saved_models/video_model_weights.npz')
fuw = np.load('saved_models/fusion_model_weights.npz')

print("=== WEIGHT NORMS ===")
print(f"FaceNet W_feat:  {np.linalg.norm(fw['W_feat']):.4f}")
print(f"FaceNet W_cls:   {np.linalg.norm(fw['W_cls']):.4f}")
print(f"PoseNet W1:      {np.linalg.norm(pw['W1']):.4f}")
print(f"PoseNet W2:      {np.linalg.norm(pw['W2']):.4f}")
print(f"VideoNet W_x:    {np.linalg.norm(vw['W_x']):.4f}")
print(f"VideoNet W_cls:  {np.linalg.norm(vw['W_cls']):.4f}")
print(f"FusionNet W_attn:{np.linalg.norm(fuw['W_attn']):.4f}")
print(f"FusionNet W_cls: {np.linalg.norm(fuw['W_cls']):.4f}")
print(f"FusionNet b_cls: {np.round(fuw['b_cls'],4)}")

datasets = load_all_datasets()
X_e, y_e, _ = datasets['emotion']
X_p, y_p, _ = datasets['pose']
X_u, y_u, _ = datasets['ucf_crime']

X_e_n = (X_e - fw['mean']) / fw['std']
X_p_n = (X_p - pw['mean']) / pw['std']
X_u_n = (X_u - vw['mean']) / vw['std']

face_net = FacialExpressionCNN()
face_net.set_weights(fw['W_feat'], fw['b_feat'], fw['W_cls'], fw['b_cls'])
pose_net = BodyPoseNetwork()
pose_net.set_weights(pw['W1'], pw['b1'], pw['W2'], pw['b2'])
video_net = VideoTemporalCNNLSTM()
video_net.set_weights(vw['W_x'], vw['W_h'], vw['b_h'], vw['W_cls'], vw['b_cls'])
ctx_net = ContextMetadataNetwork()
fusion_net = ContextAwareAttentionFusion()
fusion_net.set_weights(fuw['W_attn'], fuw['b_attn'], fuw['W_cls'], fuw['b_cls'])

print("\n=== BRANCH MODEL OUTPUTS (TRAINING DATA) ===")
fo = face_net.forward(X_e_n[:5])
EMOT = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
print("Face outputs (5 samples):")
for i in range(5):
    print(f"  true={EMOT[y_e[i]]}, pred={EMOT[np.argmax(fo['emotion_probs'][i])]}, conf={fo['confidence'][i]:.3f}")

POSE = ['Standing','Bending','Collapsed_Fall','Aggressive']
po = pose_net.forward(X_p_n[:5])
print("\nPose outputs (5 samples):")
for i in range(5):
    print(f"  true={POSE[y_p[i]]}, pred={POSE[np.argmax(po['posture_probs'][i])]}, conf={po['confidence'][i]:.3f}")

print("\n=== FUSION ANALYSIS ===")
face_feats = face_net.forward(X_e_n[:5])['features']
pose_feats = pose_net.forward(X_p_n[:5])['features']

vid_feats = []
for i in range(5):
    vo = video_net.forward(X_u_n[i])
    vid_feats.append(vo['features'])
vid_feats = np.array(vid_feats)

ctx_meta = np.array([[1.,14.,0.8,5.,0.9],[3.,2.,0.1,20.,0.3],
                      [0.,12.,0.9,2.,0.95],[4.,23.,0.15,30.,0.2],[2.,18.,0.5,10.,0.6]])
ctx_feats = ctx_net.forward(ctx_meta)['features']

CATS = ['Normal','Fall','Fighting','Panic','Loitering']
print("Direct fusion passes on training embeddings:")
for i in range(5):
    concat = np.hstack([face_feats[i], pose_feats[i], vid_feats[i], ctx_feats[i]])
    attn_logits = np.dot(concat, fuw['W_attn']) + fuw['b_attn']
    cls_logits  = np.dot(concat, fuw['W_cls'])  + fuw['b_cls']
    print(f"  Sample {i}: concat_norm={np.linalg.norm(concat):.3f}")
    print(f"    attn_logits: {np.round(attn_logits, 3)}")
    print(f"    cls_logits:  {np.round(cls_logits, 3)}")
    res = fusion_net.forward(face_feats[i], pose_feats[i], vid_feats[i], ctx_feats[i])
    print(f"    predicted:   {res['predicted_category']}, anomaly_prob={res['anomaly_probability']:.4f}, "
          f"probs={np.round(list(res['category_probs'].values()), 4)}")

print("\n=== ROOT CAUSE: CHECK TRAINING NORMALIZATION PATH ===")
print("During fusion training in train_fusion_model.py:")
print("  face_feats = face_net.forward((X_emotion[:n] - f_w['mean']) / f_w['std'])['features']")
print("  -> This is CORRECT: training uses normalized data")
print()
print("During inference in inference_engine.py:")
print("  face_feat_raw = extract_face_features(frame_np)  # returns 0..1 pixel values")
print("  face_feat_norm = (face_feat_raw - face_mean) / face_std  # normalizes pixel-space features")
print("  face_out = face_net.forward(face_feat_norm)  # produces face_out['features'] = h")
print()
print("MISMATCH: Training data was 64-dim SYNTHETIC features with class signal injected.")
print("  During inference, we feed 64 pixel intensity values (0..1).")
print("  These are NOT the same distribution!")
print("  The face model will produce garbage because it was trained on class-signal features,")
print("  not on real pixel values.")
print()
print("This means the produced embeddings from face_net are wildly different from the")
print("embeddings fusion_net was trained on. That is why fusion_net always predicts Normal.")

# Verify: what embeddings does fusion_net produce when fed with the SAME distribution as training?
print("\n=== VERIFY FUSION WITH TRAINING-DISTRIBUTION EMBEDDINGS ===")
# Use actual training data features
n = 20
face_tr = face_net.forward(X_e_n[:n])['features']  # (n, 64)
pose_tr = pose_net.forward(X_p_n[:n])['features']  # (n, 64)
vid_tr  = np.array([video_net.forward(X_u_n[i])['features'] for i in range(n)])  # (n, 64)
ctx_tr  = ctx_net.forward(ctx_meta[[0,1,2,3,4]*4][:n])['features']  # (n, 32)

correct = 0
for i in range(n):
    res = fusion_net.forward(face_tr[i], pose_tr[i], vid_tr[i], ctx_tr[i])
    is_anom = res['anomaly_probability'] > 0.5
    expected_anom = y_u[i] != 0
    correct += int(is_anom == expected_anom)
    print(f"  [{i:02d}] y_ucf={y_u[i]}({['Norm','Assl','Rob','Pan','Van'][y_u[i]]}), "
          f"pred={res['predicted_category']}, p_anomaly={res['anomaly_probability']:.3f}, "
          f"match={is_anom == expected_anom}")

print(f"\nFusion accuracy on training embeddings: {correct}/{n} = {100*correct/n:.1f}%")
print("If this is ~50% (random), the fusion model is not learning the mapping.")
print("If this is >60%, the fusion model works but the feature extractor is the bottleneck.")
