import cv2
import threading
import time
import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from buffer import ShadowReplay
from reporter import ReportProcessor


class HomeEdgeBackend:
    # TODO: CONNECT ML DETECTOR BACKEND
    def __init__(self, session_state):
        """Initialize the HomeEdge application and start recording."""
        self.st_state = session_state
        self.recorder = ShadowReplay(record_seconds=self.st_state.storage_settings['buffer_duration'] * 60)
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
        """
        Continuously monitors the video and audio buffers for motion and sound spikes.
        If a threat is detected, it saves a replay and generates a report.
        """
        # --- Detection Parameters (tune these values as needed) ---
        motion_threshold = 5000     # The amount of pixel change to trigger motion detection.
        sound_threshold = 800       # The RMS audio level to trigger sound detection.
        detection_cooldown = 15.0   # Seconds to wait after a detection before re-arming.

        # --- Initial State ---
        previous_frame = None
        last_detection_time = 0
        avg_frame = None

        print("[Detection Loop] Waiting for buffers to populate...")
        # Wait until the video buffer has at least a few frames to compare
        while len(self.recorder.video_buffer) < 10:
            if not self.recorder.is_running:
                return # Exit if recording stops
            time.sleep(0.5)
        print("[Detection Loop] Buffers ready. Starting analysis.")

        while self.st_state.detection_running:
            time.sleep(0.1)  # Process frames every 100ms

            # Skip detection if we are in the cooldown period
            if time.time() - last_detection_time < detection_cooldown:
                continue

            # --- Get latest data from buffers ---
            current_frame = self.recorder.get_latest_frame()

            if current_frame is None:
                continue # Wait if buffers are momentarily empty

            threat_detected = False
            detection_type = "Unknown"

            # ---------------------------------
            # 1. MOTION DETECTION LOGIC
            # ---------------------------------
            # Prepare frame for analysis
            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            # If the average frame is not initialized OR its shape has changed,
            # reset the background model.
            if avg_frame is None or avg_frame.shape != gray_frame.shape:
                print(f"[Detection Loop] Initializing/resetting background model to {gray_frame.shape}.")
                avg_frame = gray_frame.copy().astype("float")
                continue

            # Accumulate the weighted average between the current frame and previous frames
            cv2.accumulateWeighted(gray_frame, avg_frame, 0.5)
            frame_delta = cv2.absdiff(gray_frame, cv2.convertScaleAbs(avg_frame))

            # Threshold the delta image to get regions of significant change
            thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
            
            # The sum of the white pixels is our motion metric
            motion_level = np.sum(thresh)

            if motion_level > motion_threshold:
                threat_detected = True
                detection_type = "Motion Detected"

            # ---------------------------------
            # 2. SOUND LEVEL DETECTION LOGIC
            # ---------------------------------
            # if not threat_detected: # Only check for sound if motion wasn't already found
            try:
                audio_chunk = self.recorder.audio_buffer[-1] if self.recorder.audio_buffer else None
                if audio_chunk is None:
                    raise ValueError("Audio buffer is empty")
                # Convert audio chunk from bytes to a numpy array
                audio_samples = np.frombuffer(audio_chunk, dtype=np.int16)
                # Calculate Root Mean Square (RMS) to measure volume
                rms = np.sqrt(np.mean(audio_samples.astype(np.float64)**2))

                if rms > sound_threshold:
                    threat_detected = True
                    detection_type = "Loud Noise Detected"
            except (ValueError, TypeError):
                # Ignore potential buffer/data errors
                pass


            # ---------------------------------
            # 3. TRIGGERING AND REPORTING
            # ---------------------------------
            if not threat_detected:
                continue

            print(f"[Detection] Threat detected: {detection_type}!")
            last_detection_time = time.time() # Start cooldown

            detection_data = {
                "threat_detected": True,
                "threat_type": detection_type,
                "confidence": 0.95, # Confidence can be made dynamic later
                "performance_metrics": {},
            }

            # Save the buffered recording
            self.recorder.save_replay()
            time.sleep(3) # Give a moment for the save thread to start writing

            # Generate and store a report
            report = self.reporter.process(detection_data)
            self.st_state.archived_reports.insert(0, report)
            
            # Update performance metrics in UI
            self.st_state.performance_metrics = {
                "fps": 24, # Placeholder
                "latency": 50, # Placeholder
                "npu_usage": 75, # Placeholder
                "confidence": int(report.get("confidence", 0) * 100),
            }

            # Trigger a popup alert in the Streamlit UI
            self.st_state.popup_alert_data = {
                "type": detection_data.get("threat_type", "Unknown"),
                "confidence": detection_data.get("confidence", 0),
                "timestamp": report.get("timestamp"),
            }
            self.st_state.show_popup_alert = True
            st.rerun()

            # Keep only the last 100 reports
            if len(self.st_state.archived_reports) > 100:
                self.st_state.archived_reports.pop()
        
        print("[Backend] Detection loop stopped.")


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
