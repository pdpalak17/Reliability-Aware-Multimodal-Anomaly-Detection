import numpy as np

class ContextMetadataNetwork:
    """
    Context & Scene Metadata Network (Dense Feed-Forward Layers).
    Processes numeric structured metadata (Zone ID, Time, Illumination, Crowd Count, Baseline Normal Score)
    to output:
    1. Context embedding vector (32-dim)
    2. Contextual anomaly risk score
    3. Context reliability confidence score C_context in [0, 1]
    """
    def __init__(self, input_dim=5, hidden_dim=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W_risk = np.random.randn(hidden_dim, 1) * 0.1
        self.b_risk = np.zeros(1)

    def set_weights(self, W1, b1, W_risk, b_risk):
        self.W1 = W1
        self.b1 = b1
        self.W_risk = W_risk
        self.b_risk = b_risk

    def forward(self, metadata):
        """
        Input: metadata array [zone_id, hour, illumination, crowd_count, baseline_norm]
        Shape: (batch_size, 5) or (5,)
        Output: dict with 'features', 'context_risk', 'confidence'
        """
        is_single = (metadata.ndim == 1)
        if is_single:
            metadata = metadata.reshape(1, -1)

        # Normalize features
        norm_meta = np.zeros_like(metadata, dtype=np.float32)
        norm_meta[:, 0] = metadata[:, 0] / 5.0    # Zone ID (0-5)
        norm_meta[:, 1] = metadata[:, 1] / 24.0   # Hour (0-24)
        norm_meta[:, 2] = metadata[:, 2]          # Illumination (0-1)
        norm_meta[:, 3] = metadata[:, 3] / 50.0   # Crowd count (0-50)
        norm_meta[:, 4] = metadata[:, 4]          # Baseline norm score (0-1)

        h = np.maximum(0, np.dot(norm_meta, self.W1) + self.b1)
        
        # Risk score calculation
        risk_logit = np.dot(h, self.W_risk) + self.b_risk
        context_risk = 1.0 / (1.0 + np.exp(-risk_logit))  # Sigmoid

        # Confidence of context data is high for structured sensors
        confidence = np.full(metadata.shape[0], 0.95, dtype=np.float32)

        if is_single:
            return {
                'features': h[0],
                'context_risk': float(context_risk[0, 0]),
                'confidence': float(confidence[0])
            }
        return {
            'features': h,
            'context_risk': context_risk.flatten(),
            'confidence': confidence
        }

if __name__ == '__main__':
    model = ContextMetadataNetwork()
    dummy_meta = np.array([2.0, 22.5, 0.2, 3.0, 0.9])
    res = model.forward(dummy_meta)
    print("[ContextBranch] Forward pass successful!")
    print("  Context features shape:", res['features'].shape)
    print("  Context risk score:", res['context_risk'])
    print("  Context Confidence:", res['confidence'])
