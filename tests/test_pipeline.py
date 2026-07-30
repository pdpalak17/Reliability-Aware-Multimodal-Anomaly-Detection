import unittest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loaders import load_all_datasets
from models.face_branch import FacialExpressionCNN
from models.pose_branch import BodyPoseNetwork
from models.video_branch import VideoTemporalCNNLSTM
from models.context_branch import ContextMetadataNetwork
from models.attention_fusion import ContextAwareAttentionFusion
from models.xai_rag_engine import XAIRAGEngine
from utils.metrics import compute_multimodal_metrics

class TestMultimodalAnomalyDetection(unittest.TestCase):

    def test_dataset_loaders(self):
        datasets = load_all_datasets()
        self.assertIn('emotion', datasets)
        self.assertIn('pose', datasets)
        self.assertIn('ucf_crime', datasets)

        X_e, y_e, c_e = datasets['emotion']
        self.assertEqual(X_e.shape[1], 64)

        X_p, y_p, c_p = datasets['pose']
        self.assertEqual(X_p.shape[1], 99)

        X_u, y_u, c_u = datasets['ucf_crime']
        self.assertEqual(X_u.shape[1], 16)
        self.assertEqual(X_u.shape[2], 128)

    def test_facial_expression_branch(self):
        model = FacialExpressionCNN()
        dummy_input = np.random.randn(64)
        res = model.forward(dummy_input)
        self.assertEqual(res['features'].shape, (64,))
        self.assertEqual(len(res['emotion_probs']), 7)
        self.assertTrue(0.0 <= res['confidence'] <= 1.0)

    def test_pose_network_branch(self):
        model = BodyPoseNetwork()
        dummy_keypoints = np.random.uniform(0, 1, 99)
        res = model.forward(dummy_keypoints)
        self.assertEqual(res['features'].shape, (64,))
        self.assertEqual(len(res['posture_probs']), 4)
        self.assertTrue(0.0 <= res['confidence'] <= 1.0)

    def test_video_temporal_branch(self):
        model = VideoTemporalCNNLSTM()
        dummy_seq = np.random.randn(16, 128)
        res = model.forward(dummy_seq)
        self.assertEqual(res['features'].shape, (64,))
        self.assertEqual(len(res['temporal_probs']), 5)
        self.assertTrue(0.0 <= res['confidence'] <= 1.0)

    def test_context_metadata_branch(self):
        model = ContextMetadataNetwork()
        dummy_meta = np.array([2.0, 14.0, 0.8, 5.0, 0.9])
        res = model.forward(dummy_meta)
        self.assertEqual(res['features'].shape, (32,))
        self.assertTrue(0.0 <= res['context_risk'] <= 1.0)
        self.assertTrue(0.0 <= res['confidence'] <= 1.0)

    def test_context_aware_attention_fusion(self):
        fusion = ContextAwareAttentionFusion()
        f_face = np.random.randn(64)
        f_pose = np.random.randn(64)
        f_video = np.random.randn(64)
        f_ctx = np.random.randn(32)
        confs = {'face': 0.1, 'pose': 0.95, 'video': 0.90, 'context': 0.95}

        res = fusion.forward(f_face, f_pose, f_video, f_ctx, confidences=confs)
        self.assertIn('attention_weights', res)
        self.assertIn('predicted_category', res)
        self.assertTrue(0.0 <= res['anomaly_probability'] <= 1.0)
        self.assertTrue(0.0 <= res['reliability_score'] <= 1.0)

    def test_xai_rag_engine(self):
        engine = XAIRAGEngine()
        mock_det = {
            'predicted_category': 'Fall',
            'anomaly_probability': 0.92,
            'reliability_score': 0.88,
            'attention_weights': {'Face': 0.05, 'Pose': 0.52, 'Video Dynamics': 0.33, 'Context': 0.10},
            'category_probs': {'Normal': 0.08, 'Fall': 0.92, 'Fighting': 0.0, 'Panic': 0.0, 'Loitering': 0.0}
        }
        res = engine.generate_rag_alert(mock_det)
        self.assertIn('alert_text', res)
        self.assertIn('RAG Incident Precedent Match', res['alert_text'])

if __name__ == '__main__':
    unittest.main()
