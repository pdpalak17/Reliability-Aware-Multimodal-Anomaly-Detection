import numpy as np

class ContextAwareAttentionFusion:
    """
    Context-Aware Multimodal Attention Fusion Network.
    Fuses feature embeddings from 4 modalities:
      1. Facial Expression (64-dim)
      2. Body Pose (64-dim)
      3. Video Dynamics (64-dim)
      4. Scene Context (32-dim)
    
    Dynamically computes Softmax Attention Weights: [a_face, a_pose, a_video, a_context]
    Outputs:
      - Attended Multimodal Feature Representation
      - Anomaly Category (Normal, Fall, Fighting, Panic, Loitering)
      - Anomaly Probability Score P(Anomaly) in [0, 1]
      - Overall Reliability Score R in [0, 1]
    """
    def __init__(self, face_dim=64, pose_dim=64, video_dim=64, context_dim=32, num_categories=5):
        self.face_dim = face_dim
        self.pose_dim = pose_dim
        self.video_dim = video_dim
        self.context_dim = context_dim
        self.total_dim = face_dim + pose_dim + video_dim + context_dim
        self.num_categories = num_categories
        self.categories = ['Normal', 'Fall', 'Fighting', 'Panic', 'Loitering']

        np.random.seed(42)
        # Attention scoring network projection weights
        self.W_attn = np.random.randn(self.total_dim, 4) * 0.1
        self.b_attn = np.zeros(4)

        # Classification network weights
        self.W_cls = np.random.randn(self.total_dim, num_categories) * 0.1
        self.b_cls = np.zeros(num_categories)

    def set_weights(self, W_attn, b_attn, W_cls, b_cls):
        self.W_attn = W_attn
        self.b_attn = b_attn
        self.W_cls = W_cls
        self.b_cls = b_cls

    def forward(self, face_feat, pose_feat, video_feat, context_feat, confidences=None):
        """
        Inputs:
          face_feat: (batch, 64) or (64,)
          pose_feat: (batch, 64) or (64,)
          video_feat: (batch, 64) or (64,)
          context_feat: (batch, 32) or (32,)
          confidences: dict with keys 'face', 'pose', 'video', 'context' (each scalar or array)
        """
        is_single = (face_feat.ndim == 1)
        if is_single:
            face_feat = face_feat.reshape(1, -1)
            pose_feat = pose_feat.reshape(1, -1)
            video_feat = video_feat.reshape(1, -1)
            context_feat = context_feat.reshape(1, -1)

        batch_size = face_feat.shape[0]

        # 1. Concatenate all 4 modality feature representations
        concatenated = np.hstack([face_feat, pose_feat, video_feat, context_feat])  # (batch, total_dim)

        # 2. Compute Attention Logits
        attn_logits = np.dot(concatenated, self.W_attn) + self.b_attn  # (batch, 4)

        # Incorporate reliability confidence gating if provided
        if confidences is not None:
            c_vec = np.column_stack([
                np.atleast_1d(confidences.get('face', 0.9)),
                np.atleast_1d(confidences.get('pose', 0.9)),
                np.atleast_1d(confidences.get('video', 0.9)),
                np.atleast_1d(confidences.get('context', 0.95))
            ])
            # Add log-confidence gating factor to attention logits
            attn_logits += np.log(np.clip(c_vec, 1e-4, 1.0))

        # Softmax Attention Weights over the 4 modalities
        exp_attn = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
        attn_weights = exp_attn / np.sum(exp_attn, axis=-1, keepdims=True)  # (batch, 4) -> [face, pose, video, context]

        # 3. Apply Attention Weights to Modality Subvectors
        a_face = attn_weights[:, 0:1]
        a_pose = attn_weights[:, 1:2]
        a_video = attn_weights[:, 2:3]
        a_context = attn_weights[:, 3:4]

        attended_face = face_feat * a_face
        attended_pose = pose_feat * a_pose
        attended_video = video_feat * a_video
        attended_context = context_feat * a_context

        attended_fused = np.hstack([attended_face, attended_pose, attended_video, attended_context])

        # 4. Anomaly Classification
        logits = np.dot(attended_fused, self.W_cls) + self.b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        category_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Anomaly Probability: Sum of non-normal categories (indices 1 to 4)
        anomaly_prob = np.sum(category_probs[:, 1:], axis=-1)

        # 5. Reliability Score: Weighted sum of per-modality confidence scores
        if confidences is not None:
            c_vec = np.column_stack([
                np.atleast_1d(confidences.get('face', 0.9)),
                np.atleast_1d(confidences.get('pose', 0.9)),
                np.atleast_1d(confidences.get('video', 0.9)),
                np.atleast_1d(confidences.get('context', 0.95))
            ])
            reliability_score = np.sum(attn_weights * c_vec, axis=-1)
        else:
            reliability_score = np.full(batch_size, 0.90)

        pred_idx = np.argmax(category_probs, axis=-1)
        pred_category = [self.categories[idx] for idx in pred_idx]

        if is_single:
            return {
                'attended_features': attended_fused[0],
                'attention_weights': {
                    'Face': float(attn_weights[0, 0]),
                    'Pose': float(attn_weights[0, 1]),
                    'Video Dynamics': float(attn_weights[0, 2]),
                    'Context': float(attn_weights[0, 3])
                },
                'category_probs': {self.categories[i]: float(category_probs[0, i]) for i in range(5)},
                'predicted_category': pred_category[0],
                'anomaly_probability': float(anomaly_prob[0]),
                'reliability_score': float(reliability_score[0])
            }

        return {
            'attended_features': attended_fused,
            'attention_weights': attn_weights,
            'category_probs': category_probs,
            'predicted_category': pred_category,
            'anomaly_probability': anomaly_prob,
            'reliability_score': reliability_score
        }

if __name__ == '__main__':
    fusion = ContextAwareAttentionFusion()
    f_face = np.random.randn(64)
    f_pose = np.random.randn(64)
    f_video = np.random.randn(64)
    f_ctx = np.random.randn(32)
    confs = {'face': 0.1, 'pose': 0.95, 'video': 0.90, 'context': 0.95}  # Face occluded!
    res = fusion.forward(f_face, f_pose, f_video, f_ctx, confidences=confs)
    print("[AttentionFusion] Forward pass successful!")
    print("  Attention Weights (Occluded Face test):", res['attention_weights'])
    print("  Predicted Category:", res['predicted_category'])
    print("  Anomaly Probability:", res['anomaly_probability'])
    print("  Reliability Score:", res['reliability_score'])
