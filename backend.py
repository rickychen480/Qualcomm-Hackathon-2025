import os
import threading
import time
from buffer import video_recorder, audio_recorder, save_replay, stop_event

class HomeEdgeApp:
    def __init__(self):
        self.recording_started = False
        self.ml_thread = None
        self.video_thread = None
        self.audio_thread = None

    def start(self):
        """Start the recording and detection loop"""
        if not self.recording_started:
            print("[Backend] Starting buffer recording...")

            # Start video and audio threads directly
            self.video_thread = threading.Thread(target=video_recorder, daemon=True)
            self.audio_thread = threading.Thread(target=audio_recorder, daemon=True)

            self.video_thread.start()
            self.audio_thread.start()

            self.recording_started = True
            print("[Backend] Recording started.")

        print("[Backend] Starting ML detection loop...")
        self.ml_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.ml_thread.start()

    def detection_loop(self):
        """Simulate detection every 15 seconds (replace with actual model)"""
        while True:
            time.sleep(15)
            print("[Detection] Threat detected!")
            self.handle_ml_detection_result({"threat_type": "intruder"})

    def handle_ml_detection_result(self, detection):
        print("[Backend] Handling detection result...")
        save_replay()
        video_path = self.find_latest_video()

        report = {
            "threat_type": detection.get("threat_type", "unknown"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_path": video_path,
            "description": f"Detected threat: {detection.get('threat_type', 'unknown')}. Immediate attention required.",
            "severity": "high",
            "actions_taken": ["Alert sent to homeowner", "Video recorded"]
        }

        print(f"[Backend] Report generated: {report}")
        return report

    def find_latest_video(self):
        """Finds the latest saved replay file"""
        files = [f for f in os.listdir() if f.startswith("replay_") and f.endswith(".mp4")]
        if not files:
            return None
        return max(files, key=os.path.getctime)

    def stop(self):
        """Gracefully stop recording threads (optional)"""
        if self.recording_started:
            print("[Backend] Stopping recording threads...")
            stop_event.set()
            if self.video_thread:
                self.video_thread.join()
            if self.audio_thread:
                self.audio_thread.join()
            self.recording_started = False
            print("[Backend] Recording stopped.")


if __name__ == "__main__":
    app = HomeEdgeApp()
    app.start()

    print("[System] HomeEdge backend is running.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        app.stop()
