import numpy as np

def compute_multimodal_metrics(y_true, y_pred_probs, y_pred_categories, reliability_scores):
    """
    Computes comprehensive evaluation metrics for multimodal anomaly detection:
    - Overall Accuracy
    - Precision, Recall, F1-Score per Category
    - Mean Reliability Index for Correct vs. Incorrect Predictions
    """
    categories = ['Normal', 'Fall', 'Fighting', 'Panic', 'Loitering']
    category_map = {cat: i for i, cat in enumerate(categories)}

    y_pred_idx = np.array([category_map.get(cat, 0) for cat in y_pred_categories])
    y_true_idx = np.array(y_true)

    accuracy = np.mean(y_pred_idx == y_true_idx)

    # Correct vs. Incorrect reliability separation
    correct_mask = (y_pred_idx == y_true_idx)
    mean_rel_correct = float(np.mean(reliability_scores[correct_mask])) if np.any(correct_mask) else 0.0
    mean_rel_incorrect = float(np.mean(reliability_scores[~correct_mask])) if np.any(~correct_mask) else 0.0

    class_metrics = {}
    for cat, idx in category_map.items():
        tp = np.sum((y_pred_idx == idx) & (y_true_idx == idx))
        fp = np.sum((y_pred_idx == idx) & (y_true_idx != idx))
        fn = np.sum((y_pred_idx != idx) & (y_true_idx == idx))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        class_metrics[cat] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }

    return {
        'overall_accuracy': float(accuracy),
        'mean_reliability_correct': mean_rel_correct,
        'mean_reliability_incorrect': mean_rel_incorrect,
        'class_metrics': class_metrics
    }

if __name__ == '__main__':
    y_t = [0, 1, 1, 2, 3, 4]
    y_p_cat = ['Normal', 'Fall', 'Fall', 'Fighting', 'Panic', 'Loitering']
    y_probs = np.random.uniform(0.5, 0.99, size=(6, 5))
    rel = np.array([0.95, 0.90, 0.88, 0.92, 0.85, 0.94])
    metrics = compute_multimodal_metrics(y_t, y_probs, y_p_cat, rel)
    print("[Metrics] Computed metrics successfully:")
    print("  Overall Accuracy:", metrics['overall_accuracy'])
    print("  Reliability on Correct:", metrics['mean_reliability_correct'])
