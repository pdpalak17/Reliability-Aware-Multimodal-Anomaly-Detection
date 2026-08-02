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

from models.inference_engine import MultimodalInferenceEngine
from models.xai_rag_engine import XAIRAGEngine
from utils.video_processor import SurveillanceVideoProcessor

# Initialize Models
engine = MultimodalInferenceEngine()
xai_rag = XAIRAGEngine()
processor = SurveillanceVideoProcessor()

def gpu_decorator(fn):
    if HAS_SPACES:
        return spaces.GPU(fn)
    return fn

@gpu_decorator
def process_surveillance_input(input_image, scenario_name, zone_id, hour, illumination, crowd_count, baseline_norm, face_occluded):
    """
    Gradio Pipeline processing image input + scene metadata with real ML inference engine.
    """
    if input_image is None:
        frame_np = np.ones((480, 640, 3), dtype=np.uint8) * 40
    else:
        frame_np = np.array(input_image)

    frame_meta = {
        'zone_id': zone_id,
        'hour': hour,
        'illumination': illumination,
        'crowd_count': crowd_count,
        'baseline_norm': baseline_norm,
        'is_occluded': face_occluded,
    }

    # Run genuine neural network inference engine
    inference_result = engine.run_inference(frame_np=frame_np, metadata=frame_meta)

    # Temperature calibration
    raw_prob = inference_result['anomaly_probability']
    raw_reliability = inference_result['reliability_score']

    def calibrate(p, temp=2.5, min_p=0.03, max_p=0.97):
        if p <= 0.0: return min_p
        if p >= 1.0: return max_p
        logit = np.log(p / (1.0 - p + 1e-9))
        cal_logit = logit / temp
        cal_p = 1.0 / (1.0 + np.exp(-cal_logit))
        return float(np.clip(cal_p, min_p, max_p))

    calc_prob = calibrate(raw_prob)
    calc_reliability = float(np.clip(raw_reliability, 0.55, 0.97))
    calc_category = inference_result['predicted_category']
    attn_weights = inference_result['attention_weights']

    # Annotated Image Output
    annotated_img, persons_data = processor.process_camera_frame_multi(
        frame_np,
        anomaly_type=calc_category,
        is_occluded=face_occluded,
        prob=calc_prob,
        reliability=calc_reliability
    )

    fusion_res = {
        'predicted_category': calc_category,
        'anomaly_probability': calc_prob,
        'reliability_score': calc_reliability,
        'attention_weights': attn_weights,
        'category_probs': inference_result['category_probs'],
    }

    # Grounded RAG Alert
    rag_out = xai_rag.generate_rag_alert(
        fusion_res,
        metadata={
            'zone': f'Zone-{zone_id}',
            'hour': hour,
            'is_occluded': face_occluded,
            'action': persons_data[0].get('action', 'Standing') if persons_data else 'Standing',
            'emotion': persons_data[0].get('emotion', 'Neutral') if persons_data else 'Neutral'
        }
    )

    category_str = f"Predicted Category: {calc_category} (Prob: {int(calc_prob*100)}%, Reliability: {int(calc_reliability*100)}%)"
    attn_text = "\n".join([f"- {k.capitalize()}: {int(v*100)}%" for k, v in attn_weights.items()])

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
        gr.Image(label="MediaPipe Annotated Surveillance Frame"),
        gr.Textbox(label="Anomaly Classification Result"),
        gr.Markdown(label="RAG Plain-Language Alert Explanation"),
        gr.Textbox(label="Modality Attention Allocation")
    ],
    title="🛡️ Reliability-Aware Contextual Multimodal Anomaly Detection",
    description="Sentinel-AI Enterprise Surveillance Platform | Powered by 4-Branch Neural Fusion Engine & ZeroGPU"
)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
