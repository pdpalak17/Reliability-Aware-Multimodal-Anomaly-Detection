import os
import sys
import numpy as np
import pandas as pd
import gradio as gr

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.dataset_loaders import load_all_datasets
from models.face_branch import FacialExpressionCNN
from models.pose_branch import BodyPoseNetwork
from models.video_branch import VideoTemporalCNNLSTM
from models.context_branch import ContextMetadataNetwork
from models.attention_fusion import ContextAwareAttentionFusion
from models.xai_rag_engine import XAIRAGEngine
from utils.video_processor import SurveillanceVideoProcessor

# Initialize Models
face_net = FacialExpressionCNN()
pose_net = BodyPoseNetwork()
video_net = VideoTemporalCNNLSTM()
context_net = ContextMetadataNetwork()
fusion_net = ContextAwareAttentionFusion()
xai_rag = XAIRAGEngine()
processor = SurveillanceVideoProcessor()

if os.path.exists("saved_models/fusion_model_weights.npz"):
    w = np.load("saved_models/fusion_model_weights.npz")
    fusion_net.set_weights(w['W_attn'], w['b_attn'], w['W_cls'], w['b_cls'])

def gpu_decorator(fn):
    if HAS_SPACES:
        return spaces.GPU(fn)
    return fn

@gpu_decorator
def process_surveillance_input(input_image, scenario_name, zone_id, hour, illumination, crowd_count, baseline_norm, face_occluded):
    """
    Gradio Pipeline function processing image input + scene metadata.
    """
    if input_image is None:
        frame_np = np.ones((480, 640, 3), dtype=np.uint8) * 40
    else:
        frame_np = np.array(input_image)

    # Determine scenario dummy representations
    if "Normal" in scenario_name:
        face_conf, pose_conf, video_conf = 0.95, 0.95, 0.90
        dummy_face = np.random.randn(64) + 0.1
        dummy_pose = np.random.uniform(0.1, 0.5, 99)
        dummy_video = np.random.randn(16, 128) * 0.2
    elif "Fall" in scenario_name:
        face_conf = 0.05 if face_occluded else 0.30
        pose_conf, video_conf = 0.96, 0.90
        dummy_face = np.random.randn(64) * 0.05
        dummy_pose = np.random.uniform(0.7, 1.0, 99)
        dummy_video = np.random.randn(16, 128) * 1.5
    elif "Fighting" in scenario_name:
        face_conf, pose_conf, video_conf = 0.85, 0.95, 0.98
        dummy_face = np.random.randn(64) + 2.0
        dummy_pose = np.random.uniform(-1.5, 1.5, 99)
        dummy_video = np.random.randn(16, 128) * 2.8
    elif "Loitering" in scenario_name:
        face_conf = 0.20 if face_occluded else 0.70
        pose_conf, video_conf = 0.85, 0.40
        dummy_face = np.random.randn(64) * 0.2
        dummy_pose = np.random.uniform(0.2, 0.4, 99)
        dummy_video = np.random.randn(16, 128) * 0.05
    else:  # Panic
        face_conf, pose_conf, video_conf = 0.75, 0.88, 0.99
        dummy_face = np.random.randn(64) + 1.5
        dummy_pose = np.random.uniform(-1.0, 1.0, 99)
        dummy_video = np.random.randn(16, 128) * 3.2

    # Model Forward Pass
    f_res = face_net.forward(dummy_face)
    p_res = pose_net.forward(dummy_pose)
    v_res = video_net.forward(dummy_video)
    meta_arr = np.array([zone_id, hour, illumination, crowd_count, baseline_norm])
    c_res = context_net.forward(meta_arr)

    confidences = {'face': face_conf, 'pose': pose_conf, 'video': video_conf, 'context': 0.95}

    fusion_res = fusion_net.forward(
        f_res['features'],
        p_res['features'],
        v_res['features'],
        c_res['features'],
        confidences=confidences
    )

    # Annotated Image Output with MediaPipe 33-landmark skeleton
    annotated_img = processor.process_camera_frame(
        frame_np,
        anomaly_type=fusion_res['predicted_category'],
        is_occluded=face_occluded,
        prob=fusion_res['anomaly_probability'],
        reliability=fusion_res['reliability_score']
    )

    # RAG Alert
    rag_out = xai_rag.generate_rag_alert(fusion_res, metadata={'zone': f'Zone-{zone_id}'})

    category_str = f"Predicted Category: {fusion_res['predicted_category']} (Prob: {int(fusion_res['anomaly_probability']*100)}%, Reliability: {int(fusion_res['reliability_score']*100)}%)"
    attn_text = "\n".join([f"- {k}: {int(v*100)}%" for k, v in fusion_res['attention_weights'].items()])

    return annotated_img, category_str, rag_out['alert_text'], attn_text

demo = gr.Interface(
    fn=process_surveillance_input,
    inputs=[
        gr.Image(label="Live Camera Frame / Input Image", type="numpy"),
        gr.Dropdown(
            choices=[
                "1. Normal Pedestrian Activity",
                "2. Sudden Fall / Collapse",
                "3. Physical Fighting / Aggression",
                "4. Night-time Server Room Loitering",
                "5. Panic / Erratic Crowd Motion"
            ],
            value="1. Normal Pedestrian Activity",
            label="Surveillance Scenario"
        ),
        gr.Slider(0, 5, value=1, step=1, label="Zone Location ID"),
        gr.Slider(0, 24, value=14, step=1, label="Time of Day (Hour)"),
        gr.Slider(0.0, 1.0, value=0.85, label="Illumination Level"),
        gr.Slider(0, 50, value=3, step=1, label="Crowd Count"),
        gr.Slider(0.0, 1.0, value=0.9, label="Baseline Normal Score"),
        gr.Checkbox(label="Simulate Face Occlusion")
    ],
    outputs=[
        gr.Image(label="MediaPipe 33-Landmark Annotated Surveillance Frame"),
        gr.Textbox(label="Anomaly Classification Result"),
        gr.Markdown(label="RAG Plain-Language Alert Explanation"),
        gr.Textbox(label="Modality Attention Allocation")
    ],
    title="🛡️ Reliability-Aware Contextual Multimodal Anomaly Detection",
    description="Woxsen University Project | Authors: Palak Dwivedi et al. | Supervisor: Dr. Uday Chandra"
)

demo.queue().launch(server_name="0.0.0.0", server_port=7860)
