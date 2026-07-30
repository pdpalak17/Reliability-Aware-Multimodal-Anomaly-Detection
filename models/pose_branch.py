import numpy as np

class BodyPoseNetwork:
    """
    Body Posture Keypoint Network.
    Processes 33 skeleton landmark points (99 raw features x, y, visibility) to output:
    1. Posture classification probabilities (4 classes: Standing, Bending, Collapsed/Fall, Aggressive)
    2. Posture bottleneck feature embedding vector (64-dim)
    3. Posture keypoint tracking confidence score C_pose in [0, 1]
    """
    def __init__(self, input_dim=99, hidden_dim=64, num_postures=4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_postures = num_postures
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_postures) * 0.1
        self.b2 = np.zeros(num_postures)

    def set_weights(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def forward(self, keypoints):
        """
        Input: keypoints array of shape (batch_size, 99) or (99,)
        Output: dict with 'features', 'posture_probs', 'confidence'
        """
        is_single = (keypoints.ndim == 1)
        if is_single:
            keypoints = keypoints.reshape(1, -1)

        # Pose representation hidden layer
        h = np.maximum(0, np.dot(keypoints, self.W1) + self.b1)
        
        # Softmax posture output
        logits = np.dot(h, self.W2) + self.b2
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Compute keypoint tracking confidence based on average visibility scores
        visibilities = keypoints[:, 2::3]  # Every 3rd element is visibility score
        avg_vis = np.mean(visibilities, axis=-1)
        confidence = np.clip(avg_vis, 0.1, 0.99)

        if is_single:
            return {
                'features': h[0],
                'posture_probs': probs[0],
                'confidence': float(confidence[0])
            }
        return {
            'features': h,
            'posture_probs': probs,
            'confidence': confidence
        }

if __name__ == '__main__':
    model = BodyPoseNetwork()
    dummy_input = np.random.uniform(0, 1, 99)
    res = model.forward(dummy_input)
    print("[PoseBranch] Forward pass successful!")
    print("  Pose features shape:", res['features'].shape)
    print("  Posture probs:", res['posture_probs'])
    print("  Pose Confidence:", res['confidence'])
