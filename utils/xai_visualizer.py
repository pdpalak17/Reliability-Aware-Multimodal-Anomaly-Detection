import numpy as np
import plotly.graph_objects as go
import plotly.express as px

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

class XAIVisualizer:
    """
    Explainable AI (XAI) Visualizer providing:
    - Grad-CAM Spatial Activation Heatmap Overlays over image/video frames
    - SHAP Feature Attribution plots per modality
    - Multimodal Attention Weight distribution charts
    """

    @staticmethod
    def apply_gradcam_overlay(frame_np, heatmap, alpha=0.5):
        """
        Overlays a normalized 2D Grad-CAM heatmap onto a video/image frame.
        Input:
          frame_np: RGB image array (H, W, 3)
          heatmap: 2D float array (H, W) normalized in [0, 1]
          alpha: blend factor
        Output:
          blended_np: RGB image array (H, W, 3) with color-mapped activation map
        """
        if frame_np is None:
            return None

        h, w = frame_np.shape[:2]

        # Resize heatmap if dimensions mismatch
        if heatmap.shape[:2] != (h, w):
            if HAS_OPENCV:
                heatmap = cv2.resize(heatmap, (w, h))
            else:
                # Basic nearest neighbor fallback
                scale_y = heatmap.shape[0] / float(h)
                scale_x = heatmap.shape[1] / float(w)
                y_idx = (np.arange(h) * scale_y).astype(int)
                x_idx = (np.arange(w) * scale_x).astype(int)
                heatmap = heatmap[np.ix_(y_idx, x_idx)]

        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))

        if HAS_OPENCV:
            # Color map JET: blue=low, red=high activation
            color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        else:
            # Fallback synthetic colormap (Red/Yellow activation overlay)
            color_heatmap = np.zeros((h, w, 3), dtype=np.uint8)
            color_heatmap[:, :, 0] = heatmap_uint8  # Red channel
            color_heatmap[:, :, 1] = np.uint8(heatmap_uint8 * 0.5)  # Green channel

        # Blend original frame with heatmap
        blended = np.float32(frame_np) * (1.0 - alpha) + np.float32(color_heatmap) * alpha
        return np.uint8(np.clip(blended, 0, 255))

    @staticmethod
    def create_shap_bar_chart(shap_scores, title="SHAP Modality Feature Importance"):
        """
        Generates a Plotly horizontal bar chart for SHAP feature attribution.
        """
        modalities = list(shap_scores.keys())
        importance_values = [shap_scores[m] for m in modalities]

        # Sort ascending for horizontal bar chart
        sorted_indices = np.argsort(importance_values)
        modalities_sorted = [modalities[i] for i in sorted_indices]
        values_sorted = [importance_values[i] for i in sorted_indices]

        colors = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444'][:len(modalities_sorted)]

        fig = go.Figure(go.Bar(
            x=values_sorted,
            y=modalities_sorted,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{v*100:.1f}%" for v in values_sorted],
            textposition='auto'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#0F172A')),
            xaxis_title="SHAP Importance Value",
            yaxis_title="Modality Branch",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#334155'),
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(gridcolor='#E2E8F0', range=[0, max(values_sorted)*1.15 if values_sorted else 1.0])
        )
        return fig

    @staticmethod
    def create_attention_radar_chart(attn_weights, title="Multimodal Attention Allocation"):
        """
        Generates a Plotly polar radar chart displaying modality attention weight allocation.
        """
        modalities = [m.capitalize() for m in attn_weights.keys()]
        weights = list(attn_weights.values())

        # Complete loop for radar graph
        r_vals = weights + [weights[0]]
        theta_vals = modalities + [modalities[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.25)',
            line=dict(color='#3B82F6', width=2),
            name='Attention Weight'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.0], gridcolor='#CBD5E1', tickfont=dict(color='#64748B')),
                angularaxis=dict(gridcolor='#CBD5E1', tickfont=dict(color='#0F172A', size=12)),
                bgcolor='rgba(241, 245, 249, 0.6)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#334155'),
            title=dict(text=title, font=dict(size=16, color='#0F172A')),
            height=300,
            margin=dict(l=40, r=40, t=40, b=30),
            showlegend=False
        )
        return fig
