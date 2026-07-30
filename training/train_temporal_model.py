import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loaders import UCFCrimeDatasetLoader
from models.video_branch import VideoTemporalCNNLSTM

def train_video_temporal_model(epochs=80, lr=0.08):
    print("=" * 60)
    print("  TRAINING VIDEO TEMPORAL CNN+LSTM MODEL ON UCF-CRIME DATASET")
    print("=" * 60)

    loader = UCFCrimeDatasetLoader()
    X, y, crime_classes = loader.load_data()

    batch_size, seq_len, input_dim = X.shape
    hidden_dim = 64

    # Normalize video sequence features
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    model = VideoTemporalCNNLSTM(input_dim=128, hidden_dim=64, num_classes=len(crime_classes))

    np.random.seed(42)
    W_x = np.random.randn(input_dim, hidden_dim) * 0.2
    W_h = np.random.randn(hidden_dim, hidden_dim) * 0.2
    b_h = np.zeros(hidden_dim)
    W_cls = np.random.randn(hidden_dim, len(crime_classes)) * 0.2
    b_cls = np.zeros(len(crime_classes))

    one_hot_y = np.eye(len(crime_classes))[y]

    best_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        # Forward recurrent pass
        h_t = np.zeros((batch_size, hidden_dim))
        for t in range(seq_len):
            x_t = X_norm[:, t, :]
            h_t = np.tanh(np.dot(x_t, W_x) + np.dot(h_t, W_h) + b_h)

        logits = np.dot(h_t, W_cls) + b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        loss = -np.mean(np.sum(one_hot_y * np.log(probs + 1e-8), axis=-1))
        acc = np.mean(np.argmax(probs, axis=-1) == y)

        if acc > best_acc:
            best_acc = acc
            best_weights = (W_x.copy(), W_h.copy(), b_h.copy(), W_cls.copy(), b_cls.copy())

        grad_logits = (probs - one_hot_y) / batch_size
        grad_W_cls = np.dot(h_t.T, grad_logits)
        grad_b_cls = np.sum(grad_logits, axis=0)

        current_lr = lr * (0.96 ** (epoch // 10))
        W_cls -= current_lr * grad_W_cls
        b_cls -= current_lr * grad_b_cls

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {loss:.4f} - Accuracy: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    W_x_b, W_h_b, b_h_b, W_cls_b, b_cls_b = best_weights
    model.set_weights(W_x_b, W_h_b, b_h_b, W_cls_b, b_cls_b)

    os.makedirs("saved_models", exist_ok=True)
    np.savez("saved_models/video_model_weights.npz", W_x=W_x_b, W_h=W_h_b, b_h=b_h_b, W_cls=W_cls_b, b_cls=b_cls_b, mean=mean, std=std)
    print(f"Successfully saved trained Video Temporal CNN+LSTM model (Best Accuracy: {best_acc*100:.2f}%)!\n")

if __name__ == '__main__':
    train_video_temporal_model()
