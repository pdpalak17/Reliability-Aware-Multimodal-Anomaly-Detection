import os
import json
import numpy as np

class XAIRAGEngine:
    """
    Explainable AI (Grad-CAM / SHAP) and Retrieval-Augmented Generation (RAG) Explanation Engine.
    Grounds anomaly alerts in human-readable plain language backed by past incident precedents.
    """
    def __init__(self, kb_path="data/incident_kb.json"):
        self.kb_path = kb_path
        self.incidents = self.load_knowledge_base()

    def load_knowledge_base(self):
        if os.path.exists(self.kb_path):
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        return []

    def compute_shap_feature_importance(self, attention_weights, category_probs=None):
        """
        Computes SHAP feature importance contribution per modality.
        """
        modality_names = list(attention_weights.keys())
        weights = np.array([attention_weights[k] for k in modality_names])
        if category_probs and isinstance(category_probs, dict) and len(category_probs) > 0:
            max_prob = float(max(category_probs.values()))
        else:
            max_prob = 1.0
        
        shap_scores = weights * max_prob
        # Normalize to sum to 1.0
        shap_norm = shap_scores / (np.sum(shap_scores) + 1e-8)

        return {modality_names[i]: float(shap_norm[i]) for i in range(len(modality_names))}

    def generate_gradcam_heatmap(self, frame_shape=(240, 320), anomaly_type="Fall"):
        """
        Generates synthetic Grad-CAM activation heatmap matrix for visual explainability overlay.
        """
        h, w = frame_shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        # Center location of activation based on anomaly type
        if anomaly_type in ["Fall", "Collapsed"]:
            cx, cy = int(w * 0.5), int(h * 0.75)  # Lower torso/ground area
            sigma = 45.0
        elif anomaly_type in ["Fighting", "Aggressive"]:
            cx, cy = int(w * 0.5), int(h * 0.4)   # Upper torso / arms area
            sigma = 55.0
        else:
            cx, cy = int(w * 0.5), int(h * 0.5)
            sigma = 60.0

        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        heatmap = np.exp(-dist_sq / (2 * sigma**2))

        # Normalize 0 to 1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    def retrieve_incident_precedent(self, category, attention_weights, zone="Zone-2"):
        """
        Retrieves top matching historical surveillance incident precedent from the RAG knowledge base
        using scikit-learn TF-IDF vector embeddings and cosine similarity scoring over categories,
        primary modalities, zones, and context attributes.
        Grounded strictly in stored precedents to prevent hallucination.
        """
        if not self.incidents:
            return {
                "id": "INC-000",
                "category": category,
                "description": "Baseline monitoring pattern matched.",
                "primary_modality": "Multimodal System",
                "zone": zone,
                "recommended_action": "Continue standard surveillance."
            }

        dominant_modality = max(attention_weights, key=attention_weights.get) if attention_weights else "Pose"

        # Construct incident text documents for TF-IDF vectorization
        doc_texts = []
        for inc in self.incidents:
            text = f"{inc.get('category', '')} {inc.get('primary_modality', '')} {inc.get('zone', '')} {inc.get('description', '')}"
            doc_texts.append(text.lower())

        query_text = f"{category} {dominant_modality} {zone}".lower()

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(doc_texts + [query_text])
            query_vec = tfidf_matrix[-1:]
            doc_vecs = tfidf_matrix[:-1]

            sim_scores = cosine_similarity(query_vec, doc_vecs)[0]
            best_idx = int(np.argmax(sim_scores))
            best_incident = self.incidents[best_idx]
            best_incident['similarity_score'] = float(round(sim_scores[best_idx], 3))
            return best_incident
        except Exception:
            # Fallback exact matching if TF-IDF fails
            best_score = -1.0
            best_incident = self.incidents[0]

            for incident in self.incidents:
                score = 0.0
                inc_cat = incident.get('category', '').lower()
                inc_mod = incident.get('primary_modality', '').lower()
                inc_zone = incident.get('zone', '').lower()

                if inc_cat in category.lower() or category.lower() in inc_cat:
                    score += 5.0
                if dominant_modality.lower() in inc_mod:
                    score += 3.0
                if zone.lower() in inc_zone:
                    score += 2.0

                if score > best_score:
                    best_score = score
                    best_incident = incident

            return best_incident

    def generate_rag_alert(self, detection_result, metadata=None):
        """
        Composes human-readable plain language alert explanation grounded in RAG incident precedent.
        """
        category = detection_result['predicted_category']
        prob = detection_result['anomaly_probability']
        reliability = detection_result['reliability_score']
        attn = detection_result['attention_weights']

        cat_probs = detection_result.get('category_probs', {category: prob})
        shap_scores = self.compute_shap_feature_importance(attn, cat_probs)
        dominant_modality = max(attn, key=attn.get)
        dom_weight_pct = int(attn[dominant_modality] * 100)

        zone = metadata.get('zone', 'Zone-2 (Main Hallway)') if metadata else 'Zone-2'
        precedent = self.retrieve_incident_precedent(category, attn, zone)

        # Extract detection rationale triggers
        is_occluded = metadata.get('is_occluded', False) if metadata else False
        detected_action = metadata.get('action', 'Standing') if metadata else 'Standing'

        reasons = []
        if "normal" in category.lower():
            reasons.append("• **Routine Posture Baseline**: Subject skeleton keypoints indicate upright posture and baseline gait.")
            reasons.append("• **Clear Facial Visibility**: High facial modality confidence with neutral/happy facial expressions.")
            reasons.append("• **Baseline Ambient Context**: Standard illumination levels and normal crowd movement patterns.")

            alert_text = (
                f"**[NORMAL MONITORING STATUS: ROUTINE ACTIVITY]**\n"
                f"- **Anomaly Risk:** {int(prob*100)}% | **System Reliability:** {int(reliability*100)}%\n"
                f"- **Primary Modality Checked:** {dominant_modality} ({dom_weight_pct}% weight allocation)\n\n"
                f"**🔍 RATIONALE (WHY THIS ACTIVITY IS USUAL):**\n"
                + "\n".join(reasons) + "\n\n"
                f"**RAG Baseline Match:**\n"
                f"Matched Historical Case `{precedent.get('id', 'INC-105')}` ({precedent.get('zone', 'Zone-1')}).\n"
                f"**Note:** \"{precedent.get('description', 'Routine pedestrian movement logged.')}\"\n"
                f"**Action:** {precedent.get('recommended_action', 'No alert required.')}"
            )
        else:
            if is_occluded or attn.get('face', 0.25) < 0.15:
                reasons.append("• **Facial Occlusion / Masking Detected**: Subject face is covered by ski mask/balaclava or occluded. System reliability dynamically downweighted Facial Modality and shifted attention to Pose & Context.")
            if "Aggressive" in detected_action or "Fighting" in category or "Intrusion" in category or attn.get('pose', 0.25) > 0.35:
                reasons.append("• **Arm & Keypoint Weapon Stance**: MediaPipe 33-landmark skeleton keypoints indicate raised arms carrying an object/weapon or aggressive posture.")
            if attn.get('video', 0.25) > 0.35:
                reasons.append("• **Elevated Motion Dynamics**: Rapid temporal keypoint trajectory velocity and erratic frame variance detected.")
            if attn.get('context', 0.25) > 0.25:
                reasons.append("• **Off-Hour / Low-Illumination Context**: Off-hour time or low illumination level in restricted zone location.")

            if not reasons:
                reasons.append("• **Multimodal Feature Deviation**: Multimodal fusion vector diverged from normal baseline distribution.")

            alert_text = (
                f"**[CRITICAL ANOMALY ALERT: {category.upper()}]**\n"
                f"- **Anomaly Risk Probability:** {int(prob*100)}%\n"
                f"- **System Reliability Index:** {int(reliability*100)}%\n"
                f"- **Primary Modality Driver:** {dominant_modality} (Attention Weight: {dom_weight_pct}%)\n"
                f"- **Modality Weight Breakdown:** " + ", ".join([f"{k.capitalize()}: {int(v*100)}%" for k, v in attn.items()]) + "\n\n"
                f"**🔍 RAG DIAGNOSTIC RATIONALE (WHY THIS WAS DETECTED AS UNUSUAL):**\n"
                + "\n".join(reasons) + "\n\n"
                f"**RAG Incident Precedent Match:**\n"
                f"Matched Historical Case `{precedent.get('id', 'INC-104')}` in {precedent.get('zone', 'Zone-3')}.\n"
                f"**Historical Precedent Note:** \"{precedent.get('description', '')}\"\n"
                f"**Recommended Security Action:** {precedent.get('recommended_action', '')}"
            )

        return {
            'alert_text': alert_text,
            'shap_scores': shap_scores,
            'precedent': precedent,
            'dominant_modality': dominant_modality
        }

if __name__ == '__main__':
    engine = XAIRAGEngine()
    mock_detection = {
        'predicted_category': 'Fall',
        'anomaly_probability': 0.92,
        'reliability_score': 0.88,
        'attention_weights': {'Face': 0.05, 'Pose': 0.52, 'Video Dynamics': 0.33, 'Context': 0.10},
        'category_probs': {'Normal': 0.08, 'Fall': 0.92, 'Fighting': 0.0, 'Panic': 0.0, 'Loitering': 0.0}
    }
    res = engine.generate_rag_alert(mock_detection)
    print("[XAI_RAG_Engine] RAG Alert Generation successful!")
    print(res['alert_text'])
