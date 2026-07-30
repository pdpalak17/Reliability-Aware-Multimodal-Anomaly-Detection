import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loaders import MPIIHumanPoseDatasetLoader
from models.pose_branch import BodyPoseNetwork

def train_body_pose_model(epochs=80, lr=0.1):
    print("=" * 60)
    print("  TRAINING BODY POSE MODEL ON MPII HUMAN POSE DATASET")
    print("=" * 60)

    loader = MPIIHumanPoseDatasetLoader()
    X, y, posture_classes = loader.load_data()

    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    model = BodyPoseNetwork(input_dim=99, hidden_dim=64, num_postures=len(posture_classes))

    np.random.seed(42)
    W1 = np.random.randn(99, 64) * 0.2
    b1 = np.zeros(64)
    W2 = np.random.randn(64, len(posture_classes)) * 0.2
    b2 = np.zeros(len(posture_classes))

    one_hot_y = np.eye(len(posture_classes))[y]

    best_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        h = np.maximum(0, np.dot(X_norm, W1) + b1)
        logits = np.dot(h, W2) + b2
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        loss = -np.mean(np.sum(one_hot_y * np.log(probs + 1e-8), axis=-1))
        acc = np.mean(np.argmax(probs, axis=-1) == y)

        if acc > best_acc:
            best_acc = acc
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

        grad_logits = (probs - one_hot_y) / len(X)
        grad_W2 = np.dot(h.T, grad_logits)
        grad_b2 = np.sum(grad_logits, axis=0)

        grad_h = np.dot(grad_logits, W2.T) * (h > 0)
        grad_W1 = np.dot(X_norm.T, grad_h)
        grad_b1 = np.sum(grad_h, axis=0)

        current_lr = lr * (0.96 ** (epoch // 10))
        W2 -= current_lr * grad_W2
        b2 -= current_lr * grad_b2
        W1 -= current_lr * grad_W1
        b1 -= current_lr * grad_b1

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {loss:.4f} - Accuracy: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    W1_b, b1_b, W2_b, b2_b = best_weights
    model.set_weights(W1_b, b1_b, W2_b, b2_b)

    os.makedirs("saved_models", exist_ok=True)
    np.savez("saved_models/pose_model_weights.npz", W1=W1_b, b1=b1_b, W2=W2_b, b2=b2_b, mean=mean, std=std)
    print(f"Successfully saved trained Body Pose model (Best Accuracy: {best_acc*100:.2f}%)!\n")

if __name__ == '__main__':
    train_body_pose_model()
