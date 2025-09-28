import queue
import time
import streamlit as st
from datetime import datetime


class HomeEdgeApp:
    """Main HomeEdge Streamlit Application"""

    # TODO: CONNECT ML DETECTOR BACKEND
    def __init__(self):
        self.connect_ml_detector(None)

    # TODO: ADJUST BELOW BASED ON ACTUAL DETECTION DATA

    def connect_ml_detector(self, ml_detector_instance):
        """Connect to your ML detector backend"""
        self.ml_detector = ml_detector_instance
        self.detection_queue = queue.Queue()
        self.detection_thread = None

    def handle_ml_detection_result(self, detection_data):
        """Main method to handle ML detection results from your backend"""
        # TODO: Accept and handle ml_detector output here

        # Update performance metrics
        if "performance_metrics" in detection_data:
            st.session_state.performance_metrics.update(
                detection_data["performance_metrics"]
            )

        # Update current frame
        if "frame" in detection_data:
            st.session_state.current_frame = detection_data["frame"]

        # Handle threat detection
        if detection_data.get("threat_detected") and st.session_state.detection_running:
            self.process_threat_detection(detection_data)

        # Update audio visualization
        if (
            "audio_data" in detection_data
            and "audio_levels" in detection_data["audio_data"]
        ):
            st.session_state.audio_levels = detection_data["audio_data"]["audio_levels"]

        # Update threat level
        if detection_data.get("confidence"):
            st.session_state.threat_level = detection_data["confidence"] * 100

        # Store detection in history
        st.session_state.detection_history.append(
            {"timestamp": datetime.now(), "data": detection_data}
        )

        # Keep only last 100 detections
        if len(st.session_state.detection_history) > 100:
            st.session_state.detection_history.pop(0)


    # TODO: ADJUST + TEST BELOW BASED WITH REAL/MOCKED DETECTION DATA

    def process_threat_detection(self, detection_data):
        """Process detected threats and update alerts"""

        # Create and store alert
        alert = {
            "timestamp": datetime.now(),
            "type": detection_data["threat_type"],
            "confidence": detection_data["confidence"],
            "details": detection_data.get("bounding_boxes", []),
        }
        st.session_state.alerts.insert(0, alert)

        # Show popup alert
        st.session_state.show_popup_alert = True
        st.session_state.popup_alert_data = alert

        # Keep only last 100 alerts
        if len(st.session_state.alerts) > 100:
            st.session_state.alerts.pop()

        # Create report and save recording
        self.create_report(detection_data)

        # TODO: Render report + video

        # detection = {"threat_type": "intruder"}
        # report = st.session_state.app.handle_ml_detection_result(detection)

        # st.success("Threat detected and report generated!")

        # # Display report details
        # st.subheader("Threat Report")
        # st.write(f"**Type:** {report['threat_type']}")
        # st.write(f"**Time:** {report['timestamp']}")
        # st.write(f"**Severity:** {report['severity']}")
        # st.write(f"**Description:** {report['description']}")
        # st.write("**Actions Taken:**")
        # for action in report['actions_taken']:
        #     st.write(f"- {action}")

        # # Show video if exists
        # video_path = report.get("video_path")
        # if video_path and os.path.exists(video_path):
        #     st.video(video_path)
        # else:
        #     st.info("No video available.")

    def create_archived_report(self, detection_data):
        """Create an archived report for the threat detection"""
        recording_data = self.trigger_recording_save(detection_data)
        report = {
            "id": len(st.session_state.archived_reports) + 1,
            "timestamp": datetime.now(),
            "threat_type": detection_data["threat_type"],
            "confidence": detection_data["confidence"],
            "bounding_boxes": detection_data.get("bounding_boxes", []),
            "performance_metrics": detection_data.get("performance_metrics", {}),
            "recording_data": recording_data,
            "summary": f"{detection_data['threat_type'].title()} detection with {detection_data['confidence']:.1%} confidence",
            "status": "Active",
        }
        st.session_state.archived_reports.insert(0, report)

        # Keep only last 100 reports
        if len(st.session_state.archived_reports) > 100:
            st.session_state.archived_reports.pop()
    
    # TODO: Fill with correct data
    def trigger_recording_save(self, detection_data):
        """Trigger your storage system to save the last few minutes."""
        st.app.session_state.app.recorder.save_replay()
        recording_data = {
            "timestamp": datetime.now(),
            "threat_type": detection_data["threat_type"],
            "confidence": detection_data["confidence"],
            "duration_minutes": 3,
            "video_path": f"storage/video_{int(time.time())}.mp4",
            "audio_path": f"storage/audio_{int(time.time())}.wav",
            "size_mb": 150,
        }
        return recording_data
