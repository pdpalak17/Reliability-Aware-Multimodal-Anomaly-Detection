import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loaders import MultimodalEmotionDatasetLoader
from models.face_branch import FacialExpressionCNN

def train_facial_expression_model(epochs=80, lr=0.08):
    print("=" * 60)
    print("  TRAINING FACIAL EXPRESSION MODEL ON MULTIMODAL EMOTION DATASET")
    print("=" * 60)

    loader = MultimodalEmotionDatasetLoader()
    X, y, classes = loader.load_data()

    # Standardize input features
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    model = FacialExpressionCNN(input_dim=64, num_emotions=len(classes))
    
    np.random.seed(42)
    W_feat = np.random.randn(64, 64) * 0.2
    b_feat = np.zeros(64)
    W_cls = np.random.randn(64, len(classes)) * 0.2
    b_cls = np.zeros(len(classes))

    one_hot_y = np.eye(len(classes))[y]

    best_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        # Forward pass
        h = np.maximum(0, np.dot(X_norm, W_feat) + b_feat)
        logits = np.dot(h, W_cls) + b_cls
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        loss = -np.mean(np.sum(one_hot_y * np.log(probs + 1e-8), axis=-1))
        preds = np.argmax(probs, axis=-1)
        acc = np.mean(preds == y)

        if acc > best_acc:
            best_acc = acc
            best_weights = (W_feat.copy(), b_feat.copy(), W_cls.copy(), b_cls.copy())

        # Gradients
        grad_logits = (probs - one_hot_y) / len(X)
        grad_W_cls = np.dot(h.T, grad_logits)
        grad_b_cls = np.sum(grad_logits, axis=0)

        grad_h = np.dot(grad_logits, W_cls.T) * (h > 0)
        grad_W_feat = np.dot(X_norm.T, grad_h)
        grad_b_feat = np.sum(grad_h, axis=0)

        current_lr = lr * (0.96 ** (epoch // 10))
        W_cls -= current_lr * grad_W_cls
        b_cls -= current_lr * grad_b_cls
        W_feat -= current_lr * grad_W_feat
        b_feat -= current_lr * grad_b_feat

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {loss:.4f} - Accuracy: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    W_f, b_f, W_c, b_c = best_weights
    model.set_weights(W_f, b_f, W_c, b_c)
    
    os.makedirs("saved_models", exist_ok=True)
    np.savez("saved_models/facial_model_weights.npz", W_feat=W_f, b_feat=b_f, W_cls=W_c, b_cls=b_c, mean=mean, std=std)
    print(f"Successfully saved trained Facial Expression CNN model (Best Accuracy: {best_acc*100:.2f}%)!\n")

if __name__ == '__main__':
    train_facial_expression_model()
