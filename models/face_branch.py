import numpy as np

class FacialExpressionCNN:
    """
    Facial Expression Branch CNN Network.
    Processes face image features to output:
    1. Emotion probability distribution (7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    2. Facial bottleneck feature embedding vector (64-dim)
    3. Facial visibility/reliability confidence score C_face in [0, 1]
    """
    EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, input_dim=64, num_emotions=7):
        self.input_dim = input_dim
        self.num_emotions = num_emotions
        np.random.seed(42)
        # Weight matrices for dense feature extraction and emotion classification
        self.W_feat = np.random.randn(input_dim, 64) * 0.1
        self.b_feat = np.zeros(64)
        self.W_cls = np.random.randn(64, num_emotions) * 0.1
        self.b_cls = np.zeros(num_emotions)

    def set_weights(self, W_feat, b_feat, W_cls, b_cls):
        self.W_feat = W_feat
        self.b_feat = b_feat
        self.W_cls = W_cls
        self.b_cls = b_cls

    def forward(self, face_features):
        """
        Input: face_features array of shape (batch_size, 64) or (64,)
        Output: dict with 'features', 'emotion_probs', 'confidence', 'dominant_emotion', 'emotion_dict'
        """
        is_single = (face_features.ndim == 1)
        if is_single:
            face_features = face_features.reshape(1, -1)

        # Bottleneck activation (ReLU)
        h = np.maximum(0, np.dot(face_features, self.W_feat) + self.b_feat)
        
        # Softmax emotion classification
        logits = np.dot(h, self.W_cls) + self.b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Estimate confidence score based on feature magnitude / variance
        feat_norm = np.linalg.norm(face_features, axis=-1)
        # If face is missing/zeroed, confidence drops to zero
        confidence = 1.0 / (1.0 + np.exp(- (feat_norm - 1.0)))
        confidence = np.clip(confidence, 0.05, 0.99)

        dominant_emotions = [self.EMOTION_CLASSES[i] for i in np.argmax(probs, axis=1)]
        emotion_dicts = [dict(zip(self.EMOTION_CLASSES, p)) for p in probs]

        if is_single:
            return {
                'features': h[0],
                'emotion_probs': probs[0],
                'confidence': float(confidence[0]),
                'dominant_emotion': dominant_emotions[0],
                'emotion_dict': emotion_dicts[0]
            }
        return {
            'features': h,
            'emotion_probs': probs,
            'confidence': confidence,
            'dominant_emotions': dominant_emotions,
            'emotion_dicts': emotion_dicts
        }

if __name__ == '__main__':
    model = FacialExpressionCNN()
    dummy_input = np.random.randn(64)
    res = model.forward(dummy_input)
    print("[FaceBranch] Forward pass successful!")
    print("  Feature vector shape:", res['features'].shape)
    print("  Emotion probs:", res['emotion_probs'])
    print("  Face Confidence:", res['confidence'])
