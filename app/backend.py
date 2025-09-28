import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx

from buffer import ShadowReplay
from reporter import ReportProcessor


class HomeEdgeBackend:
    # TODO: CONNECT ML DETECTOR BACKEND
    def __init__(self, session_state):
        """Initialize the HomeEdge application and start recording."""
        self.st_state = session_state
        self.recorder = ShadowReplay(record_seconds=self.st_state.storage_settings['buffer_duration'] * 60)
        self.ml_detector = None
        self.reporter = ReportProcessor(self.recorder, self.st_state)

    def start(self):
        """Start the recording and detection loop"""
        if not self.recorder.is_running:
            self.recorder.start()
            print("[Backend] Recording started.")

        print("[Backend] Starting ML detection loop...")
        self.ml_thread = threading.Thread(target=self.detection_loop, daemon=True)
        add_script_run_ctx(self.ml_thread)
        self.ml_thread.start()
    
    def stop(self):
        """Gracefully stop the application and recording."""
        self.recorder.stop()

    def detection_loop(self):
        """Simulate detection every 15 seconds (replace with actual model)"""
        threat_detected = False
        while True:
            # TODO: Integrate actual ML model here for threat_detected and detection_data
            # Simulate threat detection for testing
            threat_detected = True
            detection_data = {
                "threat_detected": True,
                "threat_type": "intrusion",
                "confidence": 0.92,
                "performance_metrics": {
                    "fps": 22,
                    "latency": 50,
                    "npu_usage": 80,
                    "confidence": 92,
                },
            }


            time.sleep(15)
            if not threat_detected:
                continue

            print("[Detection] Threat detected!")

            # Save recording
            self.recorder.save_replay()
            time.sleep(5)  # Wait for save to complete

            # Generate and store report
            report = self.reporter.process(detection_data)
            self.st_state.archived_reports.insert(0, report)

            # --- TRIGGER POPUP ALERT ---
            # Set state to show the popup on the next UI refresh
            self.st_state.popup_alert_data = {
                "type": detection_data.get("threat_type", "Unknown"),
                "confidence": detection_data.get("confidence", 0),
                "timestamp": report.get("timestamp"),
            }
            self.st_state.show_popup_alert = True

            # Keep only last 100 reports
            if len(self.st_state.archived_reports) > 100:
                self.st_state.archived_reports.pop()
            
            # Reset for next detection
            threat_detected = False


if __name__ == "__main__":
    app = HomeEdgeBackend()
    app.start()

    print("[System] HomeEdge backend is running.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        app.stop()
