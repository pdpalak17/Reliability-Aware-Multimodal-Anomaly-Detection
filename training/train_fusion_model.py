import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loaders import load_all_datasets
from models.face_branch import FacialExpressionCNN
from models.pose_branch import BodyPoseNetwork
from models.video_branch import VideoTemporalCNNLSTM
from models.context_branch import ContextMetadataNetwork
from models.attention_fusion import ContextAwareAttentionFusion

def train_multimodal_fusion_model(epochs=300, lr=0.15):
    print("=" * 65)
    print("  JOINT TRAINING OF CONTEXT-AWARE ATTENTION FUSION NETWORK")
    print("=" * 65)

    datasets = load_all_datasets()
    X_emotion, y_emotion, _ = datasets['emotion']
    X_pose, y_pose, _ = datasets['pose']
    X_ucf, y_ucf, _ = datasets['ucf_crime']

    n_samples = min(len(X_emotion), len(X_pose), len(X_ucf))

    # Load trained branch models
    f_w = np.load('saved_models/facial_model_weights.npz')
    p_w = np.load('saved_models/pose_model_weights.npz')
    v_w = np.load('saved_models/video_model_weights.npz')

    face_net = FacialExpressionCNN()
    face_net.set_weights(f_w['W_feat'], f_w['b_feat'], f_w['W_cls'], f_w['b_cls'])

    pose_net = BodyPoseNetwork()
    pose_net.set_weights(p_w['W1'], p_w['b1'], p_w['W2'], p_w['b2'])

    video_net = VideoTemporalCNNLSTM()
    video_net.set_weights(v_w['W_x'], v_w['W_h'], v_w['b_h'], v_w['W_cls'], v_w['b_cls'])

    context_net = ContextMetadataNetwork()

    # Extract normalized representations
    face_feats = face_net.forward((X_emotion[:n_samples] - f_w['mean']) / f_w['std'])['features']
    pose_feats = pose_net.forward((X_pose[:n_samples] - p_w['mean']) / p_w['std'])['features']
    video_feats = video_net.forward((X_ucf[:n_samples] - v_w['mean']) / v_w['std'])['features']

    np.random.seed(42)
    meta_inputs = np.column_stack([
        np.random.randint(0, 6, n_samples),
        np.random.uniform(0, 24, n_samples),
        np.random.uniform(0.1, 1.0, n_samples),
        np.random.randint(0, 40, n_samples),
        np.random.uniform(0.4, 0.95, n_samples)
    ])
    context_feats = context_net.forward(meta_inputs)['features']

    # Target category mapping
    y_multimodal = np.clip(y_ucf[:n_samples], 0, 4)
    one_hot_y = np.eye(5)[y_multimodal]

    fusion_net = ContextAwareAttentionFusion()

    total_dim = 64 + 64 + 64 + 32
    W_attn = np.random.randn(total_dim, 4) * 0.2
    b_attn = np.zeros(4)
    W_cls = np.random.randn(total_dim, 5) * 0.2
    b_cls = np.zeros(5)

    # Adam optimizer momentum trackers
    mW_cls, vW_cls = np.zeros_like(W_cls), np.zeros_like(W_cls)
    mb_cls, vb_cls = np.zeros_like(b_cls), np.zeros_like(b_cls)

    best_acc = 0.0
    best_weights = None

    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(1, epochs + 1):
        concatenated = np.hstack([face_feats, pose_feats, video_feats, context_feats])
        
        attn_logits = np.dot(concatenated, W_attn) + b_attn
        exp_attn = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
        attn_weights = exp_attn / np.sum(exp_attn, axis=-1, keepdims=True)

        attended = np.hstack([
            face_feats * attn_weights[:, 0:1],
            pose_feats * attn_weights[:, 1:2],
            video_feats * attn_weights[:, 2:3],
            context_feats * attn_weights[:, 3:4]
        ])

        logits = np.dot(attended, W_cls) + b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        loss = -np.mean(np.sum(one_hot_y * np.log(probs + 1e-8), axis=-1))
        acc = np.mean(np.argmax(probs, axis=-1) == y_multimodal)

        if acc > best_acc:
            best_acc = acc
            best_weights = (W_attn.copy(), b_attn.copy(), W_cls.copy(), b_cls.copy())

        grad_logits = (probs - one_hot_y) / n_samples
        g_W_cls = np.dot(attended.T, grad_logits)
        g_b_cls = np.sum(grad_logits, axis=0)

        # Adam momentum updates
        mW_cls = beta1 * mW_cls + (1 - beta1) * g_W_cls
        vW_cls = beta2 * vW_cls + (1 - beta2) * (g_W_cls ** 2)
        m_hat = mW_cls / (1 - beta1 ** epoch)
        v_hat = vW_cls / (1 - beta2 ** epoch)
        W_cls -= (lr / (np.sqrt(v_hat) + eps)) * m_hat

        mb_cls = beta1 * mb_cls + (1 - beta1) * g_b_cls
        vb_cls = beta2 * vb_cls + (1 - beta2) * (g_b_cls ** 2)
        mb_hat = mb_cls / (1 - beta1 ** epoch)
        vb_hat = vb_cls / (1 - beta2 ** epoch)
        b_cls -= (lr / (np.sqrt(vb_hat) + eps)) * mb_hat

        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Loss: {loss:.4f} - Accuracy: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    W_a, b_a, W_c, b_c = best_weights
    fusion_net.set_weights(W_a, b_a, W_c, b_c)

    os.makedirs("saved_models", exist_ok=True)
    np.savez("saved_models/fusion_model_weights.npz", W_attn=W_a, b_attn=b_a, W_cls=W_c, b_cls=b_c)
    print(f"Successfully saved Context-Aware Attention Fusion Network (Best Accuracy: {best_acc*100:.2f}%)!\n")

if __name__ == '__main__':
    train_multimodal_fusion_model()
