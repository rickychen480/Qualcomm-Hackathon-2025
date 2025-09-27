import os
import threading
from buffer import start_recording, save_replay
import time

class HomeEdgeApp:
    def __init__(self):
        self.recording_started = False
        self.ml_thread = None

    def start(self):
        """Start recording and detection"""
        if not self.recording_started:
            print("[Backend] Starting buffer recording...")
            start_recording()
            self.recording_started = True

        print("[Backend] Starting ML detection...")
        self.ml_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.ml_thread.start()

    def detection_loop(self):
        """Simulate detection every N seconds (replace with real detection)"""
        while True:
            time.sleep(15)
            print("[Detection] Threat detected!")
            self.handle_ml_detection_result({"threat_type": "intruder"})

    def handle_ml_detection_result(self, result):
        print("[Backend] Handling detection result...")
        save_replay()
        video_path = self.find_latest_video()
        # Generate a simple report dict
        report = {
            "threat_type": detection.get("threat_type", "unknown"),
            "timestamp": "2025-09-27 14:30:00",  # You can use datetime.now()
            "video_path": video_path,
            "description": f"Detected threat: {detection.get('threat_type', 'unknown')}. Immediate attention required.",
            "severity": "high",
            "actions_taken": ["Alert sent to homeowner", "Video recorded"]
        }
            
        # Return report for frontend use
        return report

    def find_latest_video(self):
        """Returns latest replay file path"""
        files = [f for f in os.listdir() if f.startswith("replay_") and f.endswith(".mp4")]
        if not files:
            return None
        return max(files, key=os.path.getctime)
