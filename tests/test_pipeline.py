import unittest
import numpy as np
import os
import sys
import tempfile

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
from utils.xai_visualizer import XAIVisualizer
from utils.incident_logger import IncidentLogger
from utils.system_monitor import SystemMonitor

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

    def test_xai_visualizer(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        heatmap = np.random.uniform(0, 1, (100, 100))
        blended = XAIVisualizer.apply_gradcam_overlay(frame, heatmap, alpha=0.5)
        self.assertEqual(blended.shape, (100, 100, 3))

        shap_scores = {'Face': 0.1, 'Pose': 0.5, 'Video': 0.3, 'Context': 0.1}
        fig_shap = XAIVisualizer.create_shap_bar_chart(shap_scores)
        self.assertIsNotNone(fig_shap)

        fig_radar = XAIVisualizer.create_attention_radar_chart(shap_scores)
        self.assertIsNotNone(fig_radar)

    def test_incident_logger(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            logger = IncidentLogger(log_path=tmp_path)
            evt = logger.log_incident(
                anomaly_type="Sudden Fall / Collapse",
                risk_prob=0.88,
                reliability_score=0.92,
                dominant_modality="Pose",
                zone="Zone-2",
                frame_idx=15,
                rag_explanation="Fall detected."
            )
            self.assertEqual(evt["category"], "Sudden Fall / Collapse")
            self.assertEqual(len(logger.history), 1)

            df = logger.get_history_dataframe()
            self.assertEqual(len(df), 1)

            csv_str = logger.export_csv()
            self.assertIn("Sudden Fall / Collapse", csv_str)

            json_str = logger.export_json()
            self.assertIn("EVT-", json_str)

            html_str = logger.export_html_report()
            self.assertIn("Surveillance Anomaly Report", html_str)

            logger.clear_history()
            self.assertEqual(len(logger.history), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_system_monitor(self):
        monitor = SystemMonitor()
        fps = monitor.update_fps()
        self.assertGreater(fps, 0)

        hw = monitor.get_hardware_metrics()
        self.assertIn("cpu_percent", hw)
        self.assertIn("ram_percent", hw)
        self.assertIn("ram_used_gb", hw)

        gauge = monitor.create_gauge_chart(50, "Test Gauge")
        self.assertIsNotNone(gauge)

        latency_chart = monitor.create_latency_bar_chart()
        self.assertIsNotNone(latency_chart)

if __name__ == '__main__':
    unittest.main()
