import numpy as np

class VideoTemporalCNNLSTM:
    """
    Video Dynamics CNN + LSTM Temporal Network.
    Processes frame sequences (batch_size, seq_len, 128) over time to output:
    1. Temporal crime/anomaly classification (5 classes: Normal, Assault, Robbery, Panic, Vandalism)
    2. Temporal bottleneck feature embedding vector (64-dim)
    3. Motion tracking reliability confidence score C_video in [0, 1]
    """
    def __init__(self, input_dim=128, hidden_dim=64, num_classes=5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        np.random.seed(42)
        # Recurrent / temporal weighting matrix
        self.W_x = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W_h = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_h = np.zeros(hidden_dim)
        
        self.W_cls = np.random.randn(hidden_dim, num_classes) * 0.1
        self.b_cls = np.zeros(num_classes)

    def set_weights(self, W_x, W_h, b_h, W_cls, b_cls):
        self.W_x = W_x
        self.W_h = W_h
        self.b_h = b_h
        self.W_cls = W_cls
        self.b_cls = b_cls

    def forward(self, sequence_features):
        """
        Input: sequence_features array of shape (batch_size, seq_len, 128) or (seq_len, 128)
        Output: dict with 'features', 'temporal_probs', 'confidence'
        """
        is_single = (sequence_features.ndim == 2)
        if is_single:
            sequence_features = np.expand_dims(sequence_features, axis=0)

        batch_size, seq_len, _ = sequence_features.shape
        h_t = np.zeros((batch_size, self.hidden_dim))

        # Recurrent temporal processing over sequence frames
        for t in range(seq_len):
            x_t = sequence_features[:, t, :]
            h_t = np.tanh(np.dot(x_t, self.W_x) + np.dot(h_t, self.W_h) + self.b_h)

        # Classification logits
        logits = np.dot(h_t, self.W_cls) + self.b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Motion confidence score based on temporal feature stability across frames
        motion_var = np.var(sequence_features, axis=1).mean(axis=-1)
        confidence = 1.0 / (1.0 + np.exp(-motion_var))
        confidence = np.clip(confidence, 0.2, 0.98)

        if is_single:
            return {
                'features': h_t[0],
                'temporal_probs': probs[0],
                'confidence': float(confidence[0])
            }
        return {
            'features': h_t,
            'temporal_probs': probs,
            'confidence': confidence
        }

if __name__ == '__main__':
    model = VideoTemporalCNNLSTM()
    dummy_seq = np.random.randn(16, 128)
    res = model.forward(dummy_seq)
    print("[VideoBranch] Forward pass successful!")
    print("  Temporal feature shape:", res['features'].shape)
    print("  Temporal probs:", res['temporal_probs'])
    print("  Video Confidence:", res['confidence'])
