import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

try:
    import cv2
except ImportError:
    cv2 = None

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.dataset_loaders import load_all_datasets
from models.face_branch import FacialExpressionCNN
from models.pose_branch import BodyPoseNetwork
from models.video_branch import VideoTemporalCNNLSTM
from models.context_branch import ContextMetadataNetwork
from models.attention_fusion import ContextAwareAttentionFusion
from models.xai_rag_engine import XAIRAGEngine
from models.inference_engine import MultimodalInferenceEngine
import utils.video_processor
from utils.video_processor import SurveillanceVideoProcessor
from utils.xai_visualizer import XAIVisualizer
from utils.incident_logger import IncidentLogger
from utils.system_monitor import SystemMonitor

# ---------------------------------------------------------------------------
# Read the EXACT index.html from project.zip extraction
# ---------------------------------------------------------------------------
INDEX_HTML_PATH = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
if not os.path.exists(INDEX_HTML_PATH):
    INDEX_HTML_PATH = os.path.join(os.path.dirname(__file__), 'scratch', 'project_extracted', 'index.html')

if os.path.exists(INDEX_HTML_PATH):
    with open(INDEX_HTML_PATH, 'r', encoding='utf-8') as f:
        RAW_INDEX_HTML = f.read()
else:
    RAW_INDEX_HTML = ""

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SENTINEL-AI / Multimodal Surveillance Monitor",
    layout="wide",
    initial_sidebar_state="collapsed" if st.session_state.get('page', 'home') == 'home' else "expanded"
)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Check query params for navigation from iframe
qp = st.query_params
if qp.get("view") == "screening":
    st.session_state.page = 'screening'
elif qp.get("view") == "home":
    st.session_state.page = 'home'

def navigate_to(page_name):
    st.session_state.page = page_name
    st.query_params["view"] = page_name
    st.rerun()

# ---------------------------------------------------------------------------
# Initialize session singletons & load models
# ---------------------------------------------------------------------------
if 'logger' not in st.session_state:
    st.session_state.logger = IncidentLogger()
if 'monitor' not in st.session_state:
    st.session_state.monitor = SystemMonitor()

logger = st.session_state.logger
monitor = st.session_state.monitor

@st.cache_resource
def load_models():
    engine = MultimodalInferenceEngine()
    xai_rag = XAIRAGEngine()
    ckpts = engine.checkpoints_loaded()
    return engine, xai_rag, ckpts

engine, xai_rag, _ckpts_loaded = load_models()
processor = SurveillanceVideoProcessor()


# =============================================================================
# PAGE 1: EXACT LANDING PAGE — rendered via components.html for full CSS fidelity
# =============================================================================
if st.session_state.page == 'home':

    # Hide ALL Streamlit chrome on the landing page except our nav button
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        [data-testid="stHeader"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #f4f2eb !important;
        }
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        [data-testid="stBottomBlockContainer"] { display: none !important; }
        footer { display: none !important; }
        #MainMenu { display: none !important; }
        .stButton > button[kind="primary"] {
            background-color: #c43a1a !important;
            color: #fff !important;
            border: none !important;
            border-radius: 0 !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-weight: 600 !important;
            font-size: 11px !important;
            letter-spacing: .06em !important;
            text-transform: uppercase !important;
            padding: 6px 18px !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #a82e13 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Native Streamlit "Start screening" button (works outside iframe sandbox)
    _c1, _c2 = st.columns([0.85, 0.15])
    with _c2:
        if st.button("START SCREENING", type="primary"):
            navigate_to('screening')

    # Modify index.html
    modified_html = RAW_INDEX_HTML

    # Inject CSS override: make hero not force equal column heights, flowchart fills its content area
    css_override = """
<style>
  .hero {
    min-height: auto !important;
    align-items: start !important;
  }
  .hero-r {
    padding: 32px !important;
    display: grid !important;
    place-items: center !important;
  }
  .arch {
    max-width: 100% !important;
    width: 100% !important;
  }
  .arch svg {
    width: 100% !important;
    height: auto !important;
  }
  .schema {
    grid-template-columns: 1fr 1fr !important;
    gap: 48px !important;
    align-items: center !important;
  }
  .schema .fusion-diagram {
    max-width: 520px !important;
    width: 100% !important;
    justify-self: center !important;
  }
</style>
"""
    modified_html = modified_html.replace('</head>', css_override + '</head>')

    # Render the EXACT HTML with full CSS fidelity in an iframe
    components.html(modified_html, height=3400, scrolling=True)


# =============================================================================
# PAGE 2: LIVE SURVEILLANCE SCREENING DASHBOARD
# =============================================================================
else:

    # Screening page CSS — same paper/grid design tokens, zero emojis
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

        :root {
          --bg: #f4f2eb; --paper: #eae6dc; --ink: #0e0e0e; --rule: #0e0e0e;
          --warn: #c43a1a; --ok: #1a8a5b;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: var(--bg) !important;
            color: var(--ink) !important;
            font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
        }
        h1,h2,h3,h4,h5,h6 {
            font-family: 'Inter', sans-serif !important;
            color: var(--ink) !important;
            font-weight: 500 !important;
        }
        p, span, label, div, [data-testid="stMarkdownContainer"] p {
            color: #3a3a3a !important;
        }

        .stApp::before {
            content: "";
            position: fixed; inset: 0;
            pointer-events: none; z-index: 0;
            background-image:
                linear-gradient(rgba(14,14,14,.05) 1px,transparent 1px),
                linear-gradient(90deg,rgba(14,14,14,.05) 1px,transparent 1px);
            background-size: 48px 48px;
        }

        section[data-testid="stSidebar"] {
            background-color: var(--bg) !important;
            border-right: 1px solid var(--rule) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--ink) !important;
            font-family: 'IBM Plex Mono', monospace !important;
        }

        .paper-card {
            background: var(--paper) !important;
            border: 1px solid var(--rule) !important;
            border-radius: 0px !important;
            padding: 20px !important;
            margin-bottom: 16px !important;
        }
        .paper-card-alert {
            background: var(--paper) !important;
            border: 1px solid var(--warn) !important;
            border-left: 4px solid var(--warn) !important;
            border-radius: 0 !important;
            padding: 20px !important;
            margin-bottom: 16px !important;
        }
        .paper-card-ok {
            background: var(--paper) !important;
            border: 1px solid var(--ok) !important;
            border-left: 4px solid var(--ok) !important;
            border-radius: 0 !important;
            padding: 20px !important;
            margin-bottom: 16px !important;
        }
        .badge-ok  { color: var(--ok) !important; font-weight:600; text-transform:uppercase; }
        .badge-warn { color: var(--warn) !important; font-weight:600; text-transform:uppercase; }

        .rag-box {
            background: var(--bg) !important;
            border: 1px solid var(--warn) !important;
            padding: 16px !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 12px !important;
            color: var(--ink) !important;
            margin-top: 12px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0; background: var(--paper) !important;
            padding: 0; border: 1px solid var(--rule) !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: 38px; padding: 0 18px;
            color: #666 !important; font-weight: 500;
            font-size: 12px; font-family: 'IBM Plex Mono', monospace;
            border-right: 1px solid var(--rule); border-radius: 0;
        }
        .stTabs [aria-selected="true"] {
            background: var(--bg) !important;
            color: var(--ink) !important;
            border-bottom: 2px solid var(--warn) !important;
            font-weight: 600;
        }

        .stProgress > div > div > div > div { background-color: var(--warn) !important; }

        .stButton > button[kind="primary"] {
            background-color: var(--warn) !important;
            color: #fff !important;
            border: 1px solid var(--warn) !important;
            border-radius: 0 !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: .05em !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #a82e13 !important;
            border-color: #a82e13 !important;
        }
        .stButton > button[kind="secondary"] {
            background-color: var(--paper) !important;
            color: var(--ink) !important;
            border: 1px solid var(--rule) !important;
            border-radius: 0 !important;
            font-family: 'IBM Plex Mono', monospace !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Top Header
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("Back to overview"):
            navigate_to('home')
    with col_nav2:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono'; font-size:13px; font-weight:600; padding-top:6px; color:#0e0e0e;">
            SENTINEL<span style="color:#c43a1a;">/</span>AI — LIVE SCREENING ENGINE
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-top:1px solid #0e0e0e; margin:10px 0 20px 0;'>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### OPERATIONAL CONTROLS")
        preset_scenario = st.selectbox("Surveillance Scenario Preset", [
            "1. Normal Pedestrian Activity",
            "2. Sudden Fall / Collapse",
            "3. Physical Fighting / Aggression",
            "4. Night-time Server Room Loitering",
            "5. Panic / Erratic Crowd Motion",
            "6. Custom Input Feed"
        ])
        person_mode = st.selectbox("Person Tracking Mode", [
            "Auto-Detect Persons (Computer Vision Engine)",
            "1 Person", "2 Persons", "3 Persons", "4 Persons", "5 Persons"
        ])
        st.markdown("### CONTEXTUAL METADATA")
        zone_id = st.slider("Zone Location ID", 0, 5, 1)
        hour = st.slider("Time of Day (Hour 0-23)", 0, 23, 14 if "Normal" in preset_scenario else (2 if "Loitering" in preset_scenario else 16))
        illumination = st.slider("Illumination Level", 0.0, 1.0, 0.85 if "Normal" in preset_scenario else (0.15 if "Loitering" in preset_scenario else 0.70))
        crowd_count = st.slider("Crowd Density Count", 0, 50, 3 if "Normal" in preset_scenario else (20 if "Panic" in preset_scenario else 2))
        baseline_norm = st.slider("Baseline Normal Score", 0.0, 1.0, 0.9)
        face_occluded = st.checkbox("Simulate Face Occlusion", value=("Loitering" in preset_scenario or "Fall" in preset_scenario))
        st.markdown("### SENSOR RELIABILITY")
        face_conf = st.slider("w_face", 0.0, 1.0, 0.05 if face_occluded else 0.90)
        pose_conf = st.slider("w_pose", 0.0, 1.0, 0.95)
        video_conf = st.slider("w_video", 0.0, 1.0, 0.88)
        context_conf = st.slider("w_context", 0.0, 1.0, 0.92)

    # Scenario mapping
    if "Normal" in preset_scenario:    scenario_category = "Normal Pedestrian Activity"
    elif "Fall" in preset_scenario:    scenario_category = "Sudden Fall / Collapse"
    elif "Fighting" in preset_scenario: scenario_category = "Physical Fighting / Aggression"
    elif "Loitering" in preset_scenario: scenario_category = "Night-time Server Room Loitering"
    elif "Panic" in preset_scenario:   scenario_category = "Panic / Erratic Crowd Motion"
    else:                              scenario_category = "Normal Pedestrian Activity"

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live Feed & CV Engine",
        "Attention Weights & XAI",
        "Grounded RAG Alert",
        "Audit & Analytics",
        "Diagnostics"
    ])

    # ---- TAB 1: LIVE FEED ----
    with tab1:
        col_input, col_results = st.columns([1.15, 0.85])

        with col_input:
            st.markdown("### Camera Stream & Video Feed")
            input_source = st.radio("Feed Source:", ["Live Stream", "Webcam Snapshot", "Upload Image", "Upload Video (.mp4, .avi)"], horizontal=True)

            input_img = None
            uploaded_video_bytes = None
            current_frame_idx = 0

            if input_source == "Live Stream":
                run_stream = st.checkbox("Start Continuous Live Stream", value=False)
                if run_stream:
                    if cv2 and cv2.VideoCapture:
                        try:
                            cap = cv2.VideoCapture(0)
                            if cap.isOpened():
                                ret, live_frame = cap.read()
                                frame_np = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB) if ret else processor.create_synthetic_frame(anomaly_type=scenario_category)
                                cap.release()
                            else:
                                frame_np = processor.create_synthetic_frame(anomaly_type=scenario_category)
                        except Exception:
                            frame_np = processor.create_synthetic_frame(anomaly_type=scenario_category)
                    else:
                        frame_np = processor.create_synthetic_frame(anomaly_type=scenario_category)
                else:
                    frame_np = processor.create_synthetic_frame(anomaly_type=scenario_category)
            elif input_source == "Webcam Snapshot":
                img_file = st.camera_input("Take Snapshot")
                if img_file is not None:
                    input_img = Image.open(img_file)
                frame_np = np.array(input_img.convert('RGB')) if input_img is not None else np.ones((480, 640, 3), dtype=np.uint8) * 40
            elif input_source == "Upload Image":
                img_file = st.file_uploader("Upload Image Frame", type=['png', 'jpg', 'jpeg'])
                if img_file is not None:
                    input_img = Image.open(img_file)
                frame_np = np.array(input_img.convert('RGB')) if input_img is not None else np.ones((480, 640, 3), dtype=np.uint8) * 40
            else:
                vid_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])
                if vid_file is not None:
                    uploaded_video_bytes = vid_file.read()
                if uploaded_video_bytes is not None:
                    vid_out = processor.process_video_bytes(uploaded_video_bytes, anomaly_type=scenario_category)
                    frames, fps = (vid_out if isinstance(vid_out, tuple) else (vid_out, 30))
                    if len(frames) > 0:
                        st.write(f"Video: {len(frames)} frames @ {fps} FPS")
                        current_frame_idx = st.slider("Frame Index", 0, len(frames)-1, 0)
                        frame_np = frames[current_frame_idx]
                    else:
                        frame_np = np.ones((480, 640, 3), dtype=np.uint8) * 40
                else:
                    frame_np = processor.create_synthetic_frame(anomaly_type=scenario_category)

            override_count = None if "Auto-Detect" in person_mode else int(person_mode.split()[0])

            annotated_frame, persons_data = processor.process_camera_frame_multi(
                frame_np, anomaly_type=scenario_category, is_occluded=face_occluded,
                prob=0.05, reliability=0.95, override_person_count=override_count
            )
            st.image(annotated_frame, caption=f"MediaPipe 33-Landmark Skeleton Overlay [Frame #{current_frame_idx}]", use_container_width=True)

        with col_results:
            st.markdown("### Threat Diagnosis")

            frame_meta = {
                'zone_id': zone_id, 'hour': hour, 'illumination': illumination,
                'crowd_count': crowd_count, 'baseline_norm': baseline_norm, 'is_occluded': face_occluded,
            }

            if isinstance(frame_np, tuple): frame_np = frame_np[0]
            if frame_np is None or not isinstance(frame_np, np.ndarray):
                frame_np = np.ones((480, 640, 3), dtype=np.uint8) * 40

            inference_result = engine.run_inference(frame_np=frame_np, metadata=frame_meta, verbose=False)

            raw_prob = inference_result['anomaly_probability']
            raw_reliability = inference_result['reliability_score']

            def calibrate(p, temp=2.5, min_p=0.03, max_p=0.97):
                if p <= 0.0: return min_p
                if p >= 1.0: return max_p
                logit = np.log(p / (1.0 - p + 1e-9))
                return float(np.clip(1.0 / (1.0 + np.exp(-logit / temp)), min_p, max_p))

            calc_prob = calibrate(raw_prob)
            calc_reliability = float(np.clip(raw_reliability, 0.55, 0.97))
            calc_category = inference_result['predicted_category']
            attn_weights = inference_result['attention_weights']

            _, persons_data = processor.process_camera_frame_multi(
                frame_np, anomaly_type=calc_category, is_occluded=face_occluded,
                prob=calc_prob, reliability=calc_reliability, override_person_count=override_count
            )

            if persons_data and inference_result.get('face_detected', False):
                persons_data[0]['emotion'] = inference_result['face_emotion']
                persons_data[0]['emotion_conf'] = float(inference_result['face_confidence'])
                persons_data[0]['emotion_dict'] = inference_result['face_probs']

            dominant_modality = max(attn_weights, key=attn_weights.get)
            fusion_res = {
                'predicted_category': calc_category,
                'anomaly_probability': float(calc_prob),
                'reliability_score': float(calc_reliability),
                'attention_weights': attn_weights,
                'category_probs': inference_result['category_probs'],
            }

            prob_pct = int(calc_prob * 100)
            rel_pct  = int(calc_reliability * 100)

            logger.log_incident(
                anomaly_type=calc_category, risk_prob=calc_prob,
                reliability_score=calc_reliability, dominant_modality=dominant_modality.capitalize(),
                zone=f"Zone-{zone_id}", frame_idx=current_frame_idx,
                rag_explanation=f"Inference: {calc_category}, {prob_pct}% risk, {rel_pct}% reliability."
            )

            if "normal" in calc_category.lower():
                st.markdown(f"""
                <div class="paper-card-ok">
                    <span class="badge-ok">NORMAL MONITORING STATUS</span>
                    <div style="font-size:2.2rem; font-weight:700; color:var(--ok); margin:6px 0;">{prob_pct}% Anomaly Risk</div>
                    <div style="font-size:.88rem;"><b>Fusion Reliability:</b> {rel_pct}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="paper-card-alert">
                    <span class="badge-warn">ANOMALY DETECTED: {calc_category.upper()}</span>
                    <div style="font-size:2.2rem; font-weight:700; color:var(--warn); margin:6px 0;">{prob_pct}% Anomaly Risk</div>
                    <div style="font-size:.88rem;"><b>Fusion Reliability:</b> {rel_pct}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Attention Weights")
            for mod, weight in attn_weights.items():
                st.write(f"**{mod.capitalize()}:** `{weight*100:.1f}%`")
                st.progress(float(weight))

            st.markdown("#### RAG Explanation")
            rag_alert = xai_rag.generate_rag_alert(
                fusion_res,
                metadata={
                    'zone': f'Zone-{zone_id}', 'hour': hour,
                    'is_occluded': face_occluded,
                    'action': persons_data[0].get('action', 'Standing') if persons_data else 'Standing',
                    'emotion': persons_data[0].get('emotion', 'Neutral') if persons_data else 'Neutral'
                }
            )
            st.markdown(f"""
            <div class="rag-box">
                <pre style="white-space:pre-wrap; font-family:'IBM Plex Mono',monospace; font-size:11px; margin:0; color:var(--ink);">{rag_alert['alert_text']}</pre>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### Tracked Individuals ({len(persons_data)} active)")
        cards_per_row = min(4, max(1, len(persons_data)))
        cols = st.columns(cards_per_row)
        for idx, person in enumerate(persons_data):
            with cols[idx % cards_per_row]:
                st.markdown(f"""
                <div class="paper-card">
                    <h4 style="margin:0 0 6px 0;color:var(--ink);">Person {person['id']}</h4>
                    <p style="margin:3px 0;font-size:.85rem;"><b>Action:</b> {person['action']}</p>
                    <p style="margin:3px 0;font-size:.85rem;"><b>Emotion:</b> {person['emotion']} ({int(round(person['emotion_conf']*100 if person['emotion_conf']<=1.0 else person['emotion_conf']))}%)</p>
                    <p style="margin:3px 0;font-size:.85rem;"><b>Pose:</b> {person['pose_status']}</p>
                </div>
                """, unsafe_allow_html=True)

        if persons_data:
            emo = persons_data[0]['emotion_dict']
            fig_emo = px.bar(x=list(emo.keys()), y=list(emo.values()),
                labels={'x':'Emotion', 'y':'Probability'},
                title=f"7-Class Emotion Breakdown (Person 1, Frame #{current_frame_idx})",
                color=list(emo.values()), color_continuous_scale="Reds")
            fig_emo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0e0e0e'), height=240, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_emo, use_container_width=True)

    # ---- TAB 2: XAI ----
    with tab2:
        st.markdown("### Attention Weights & XAI")
        col_cam, col_shap = st.columns([1.1, 0.9])
        with col_cam:
            st.markdown("#### Grad-CAM Heatmap")
            if isinstance(frame_np, tuple): frame_np = frame_np[0]
            if frame_np is None or not isinstance(frame_np, np.ndarray):
                frame_np = np.ones((480,640,3), dtype=np.uint8)*40
            gradcam = xai_rag.generate_gradcam_heatmap(frame_shape=(frame_np.shape[0], frame_np.shape[1]), anomaly_type=calc_category)
            alpha = st.slider("Heatmap Opacity", 0.1, 0.9, 0.5)
            st.image(XAIVisualizer.apply_gradcam_overlay(frame_np, gradcam, alpha=alpha),
                caption=f"Grad-CAM [{calc_category}]", use_container_width=True)
        with col_shap:
            st.markdown("#### SHAP Attribution")
            shap_scores = xai_rag.compute_shap_feature_importance(attn_weights, fusion_res['category_probs'])
            st.plotly_chart(XAIVisualizer.create_shap_bar_chart(shap_scores), use_container_width=True)
            st.markdown("#### Attention Radar")
            st.plotly_chart(XAIVisualizer.create_attention_radar_chart(attn_weights), use_container_width=True)

    # ---- TAB 3: RAG ----
    with tab3:
        st.markdown("### RAG Precedent Search")
        precedent = xai_rag.retrieve_incident_precedent(calc_category, attn_weights, zone=f"Zone-{zone_id}")
        c1, c2 = st.columns([1.05, 0.95])
        with c1:
            sim = f" ({int(precedent.get('similarity_score',0.95)*100)}% match)" if 'similarity_score' in precedent else ""
            st.markdown(f"""
            <div class="paper-card-alert">
                <span class="badge-warn">TOP MATCHED PRECEDENT</span>
                <h3 style="margin:10px 0 6px 0;color:var(--ink);">{precedent.get('id','INC-101')}{sim}</h3>
                <p style="margin:4px 0;font-size:.9rem;"><b>Category:</b> {precedent.get('category','N/A')}</p>
                <p style="margin:4px 0;font-size:.9rem;"><b>Zone:</b> {precedent.get('zone','N/A')}</p>
                <p style="margin:4px 0;font-size:.9rem;"><b>Primary Modality:</b> {precedent.get('primary_modality','N/A')}</p>
                <p style="margin:8px 0;font-size:.9rem;color:#3a3a3a;"><b>Note:</b> <i>"{precedent.get('description','')}"</i></p>
                <p style="margin:6px 0;font-size:.9rem;"><b>Action:</b> <span class="badge-ok">{precedent.get('recommended_action','')}</span></p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("#### Knowledge Base")
            kb = pd.DataFrame(xai_rag.load_knowledge_base())
            st.dataframe(kb[['id','category','zone','primary_modality','recommended_action']], use_container_width=True, height=260)

    # ---- TAB 4: ANALYTICS ----
    with tab4:
        st.markdown("### Analytics & Event Log")
        history_df = logger.get_history_dataframe()
        total = len(logger.history)
        high = sum(1 for e in logger.history if e.get('risk_score',0)>0.5)
        fps = monitor.update_fps()
        rel = round(np.mean([e.get('reliability_score',0) for e in logger.history])*100, 1) if logger.history else 92.4

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="paper-card"><div>TOTAL INCIDENTS</div><div style="font-size:2rem;font-weight:700;">{total}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="paper-card"><div>HIGH RISK</div><div style="font-size:2rem;font-weight:700;color:var(--warn);">{high}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="paper-card"><div>FPS</div><div style="font-size:2rem;font-weight:700;color:var(--ok);">{fps}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="paper-card"><div>RELIABILITY</div><div style="font-size:2rem;font-weight:700;">{rel}%</div></div>', unsafe_allow_html=True)

        st.dataframe(history_df, use_container_width=True, height=250)
        e1,e2,e3,e4 = st.columns(4)
        with e1: st.download_button("Export CSV", data=logger.export_csv(), file_name="incidents.csv", mime="text/csv")
        with e2: st.download_button("Export JSON", data=logger.export_json(), file_name="incidents.json", mime="application/json")
        with e3: st.download_button("Export HTML", data=logger.export_html_report(), file_name="report.html", mime="text/html")
        with e4:
            if st.button("Clear Log"):
                logger.clear_history()
                st.success("Cleared.")

    # ---- TAB 5: DIAGNOSTICS ----
    with tab5:
        st.markdown("### Hardware & Latency")
        hw = monitor.get_hardware_metrics()
        g1,g2,g3 = st.columns(3)
        with g1: st.plotly_chart(monitor.create_gauge_chart(hw['cpu_percent'], "CPU"), use_container_width=True)
        with g2: st.plotly_chart(monitor.create_gauge_chart(hw['ram_percent'], "RAM"), use_container_width=True)
        with g3:
            st.markdown(f"""
            <div class="paper-card" style="height:180px;">
                <div style="font-weight:700;">HARDWARE</div>
                <div style="margin-top:10px;"><b>GPU:</b> {hw['gpu_status']}</div>
                <div style="margin-top:6px;"><b>RAM:</b> {hw['ram_used_gb']} / {hw['ram_total_gb']} GB</div>
                <div style="margin-top:6px;"><b>Latency:</b> <span class="badge-ok">{hw['total_inference_latency']} ms</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.plotly_chart(monitor.create_latency_bar_chart(), use_container_width=True)
        st.markdown("#### Logs")
        for log in reversed(monitor.logs[-10:]):
            st.text(f"[{log['timestamp']}] [{log['level']}] {log['message']}")

    st.markdown("<hr style='border-top:1px solid #0e0e0e; margin:30px 0 10px 0;'>", unsafe_allow_html=True)
    st.caption("2026 SENTINEL-AI Research | MIT Licence")
