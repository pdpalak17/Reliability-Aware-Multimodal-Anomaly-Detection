import os
import json
import time
import pandas as pd
from datetime import datetime

class IncidentLogger:
    """
    Surveillance Incident Event Logger & Exporter.
    Logs each detected anomaly with timestamp, frame index, risk, reliability score,
    primary modality driver, zone location, and RAG explanation.
    Provides export options for CSV, JSON, and formatted HTML/Markdown reports.
    """
    def __init__(self, log_path="data/incident_history.json"):
        self.log_path = log_path
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_history(self):
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"[IncidentLogger Error] Failed to save history: {e}")

    def log_incident(self, anomaly_type, risk_prob, reliability_score, dominant_modality, zone, frame_idx=0, rag_explanation="", metadata=None):
        """
        Logs a new incident event.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_id = f"EVT-{int(time.time()*1000) % 1000000}"

        event = {
            "id": event_id,
            "timestamp": timestamp,
            "category": anomaly_type,
            "risk_score": float(round(risk_prob, 4)),
            "reliability_score": float(round(reliability_score, 4)),
            "dominant_modality": dominant_modality,
            "zone": zone,
            "frame_idx": int(frame_idx),
            "explanation": rag_explanation,
            "status": "FLAGGED" if anomaly_type != "Normal Pedestrian Activity" else "RESOLVED",
            "metadata": metadata or {}
        }

        # Avoid exact duplicate logs within 2 seconds
        if self.history:
            last_event = self.history[-1]
            if last_event["category"] == anomaly_type and abs(last_event["risk_score"] - risk_prob) < 0.02 and last_event["frame_idx"] == frame_idx:
                return last_event

        self.history.append(event)
        self.save_history()
        return event

    def get_history_dataframe(self):
        """
        Returns pandas DataFrame representation of incident history.
        """
        if not self.history:
            return pd.DataFrame(columns=[
                "ID", "Timestamp", "Category", "Risk Probability", "Reliability Index", "Primary Modality", "Zone", "Frame #", "Status"
            ])

        records = []
        for e in self.history:
            records.append({
                "ID": e.get("id", ""),
                "Timestamp": e.get("timestamp", ""),
                "Category": e.get("category", ""),
                "Risk Probability": f"{int(e.get('risk_score', 0) * 100)}%",
                "Reliability Index": f"{int(e.get('reliability_score', 0) * 100)}%",
                "Primary Modality": e.get("dominant_modality", ""),
                "Zone": e.get("zone", ""),
                "Frame #": e.get("frame_idx", 0),
                "Status": e.get("status", "")
            })
        return pd.DataFrame(records)

    def export_csv(self):
        """Returns CSV string of history."""
        df = self.get_history_dataframe()
        return df.to_csv(index=False)

    def export_json(self):
        """Returns formatted JSON string of history."""
        return json.dumps(self.history, indent=2)

    def export_html_report(self):
        """
        Generates a sleek, printable HTML report of surveillance incident logs.
        """
        df = self.get_history_dataframe()
        table_html = df.to_html(classes="styled-table", index=False)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reliability-Aware Surveillance Anomaly Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #f8fafc; color: #1e293b; }}
                h1 {{ color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
                .header-info {{ margin-bottom: 20px; background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .styled-table {{ border-collapse: collapse; margin: 25px 0; font-size: 0.9em; min-width: 100%; box-shadow: 0 0 20px rgba(0, 0, 0, 0.05); background-color: #ffffff; border-radius: 8px; overflow: hidden; }}
                .styled-table thead tr {{ background-color: #1e293b; color: #ffffff; text-align: left; font-weight: bold; }}
                .styled-table th, .styled-table td {{ padding: 12px 15px; border-bottom: 1px solid #dddddd; }}
                .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f4f6; }}
                .styled-table tbody tr:last-of-type {{ border-bottom: 2px solid #1e293b; }}
            </style>
        </head>
        <body>
            <h1>🛡️ Reliability-Aware Surveillance Anomaly Report</h1>
            <div class="header-info">
                <p><b>Report Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><b>Total Logged Events:</b> {len(self.history)}</p>
                <p><b>System:</b> Multimodal Anomaly Detection Platform v2.0</p>
            </div>
            <h2>📋 Detailed Incident Log</h2>
            {table_html}
        </body>
        </html>
        """
        return html_content

    def clear_history(self):
        self.history = []
        self.save_history()
