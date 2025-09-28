import queue
import random
import time
import streamlit as st
import strings
from datetime import datetime
from backend import HomeEdgeBackend

# Configure Streamlit page
st.set_page_config(
    page_title="HomeEdge Security Assistant",
    page_icon="Shield",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Custom CSS for better styling (keeping existing styles)
st.markdown(strings.CSS, unsafe_allow_html=True)


class HomeEdgeApp:
    """Main HomeEdge Streamlit Application"""

    # ------ SESSION STATE INITIALIZATION ------

    # TODO: CONNECT ML DETECTOR BACKEND
    def __init__(self):
        self.initialize_session_state()
        self.connect_ml_detector(None)

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""

        # Backend app instance
        if "app" not in st.session_state:
            st.session_state.app = HomeEdgeBackend()

        # Current page
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Control Dashboard"

        # State of detector (on/off)
        if "detection_running" not in st.session_state:
            st.session_state.detection_running = False

        # Threat detection history
        if "alerts" not in st.session_state:
            st.session_state.alerts = []
        if "archived_reports" not in st.session_state:
            st.session_state.archived_reports = []
        if "detection_history" not in st.session_state:
            st.session_state.detection_history = []

        # Performance metrics
        if "performance_metrics" not in st.session_state:
            st.session_state.performance_metrics = {
                "fps": 24,
                "latency": 45,
                "npu_usage": 78,
                "confidence": 85,
            }

        # Current levels
        if "current_frame" not in st.session_state:
            st.session_state.current_frame = None
        if "audio_levels" not in st.session_state:
            st.session_state.audio_levels = [
                random.uniform(0.1, 0.8) for _ in range(20)
            ]
        if "threat_level" not in st.session_state:
            st.session_state.threat_level = 15

        # Detection sensitivity settings
        if "detection_config" not in st.session_state:
            st.session_state.detection_config = {
                "person_sensitivity": 85,
                "motion_sensitivity": 70,
                "audio_sensitivity": 75,
                "sound_threshold": 60,
            }

        # Automatic Schedule Settings
        if "schedule_config" not in st.session_state:
            st.session_state.schedule_config = {
                "enable": False,
                "start_time": datetime.strptime("22:00", "%H:%M").time(),  # 10:00 PM
                "stop_time": datetime.strptime("06:00", "%H:%M").time(),  # 6:00 AM
                "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            }

        # Popup alerts
        if "show_popup_alert" not in st.session_state:
            st.session_state.show_popup_alert = False
        if "popup_alert_data" not in st.session_state:
            st.session_state.popup_alert_data = {}

    # ------ ML DETECTION HANDLING ------
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

    # ------ ML DETECTION PROCESSING ------
    # TODO: ADJUST + TEST BELOW BASED WITH REAL/MOCKED DETECTION DATA

    def process_threat_detection(self, detection_data):
        """Process detected threats and update alerts"""
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

        # Keep only last 50 alerts
        if len(st.session_state.alerts) > 50:
            st.session_state.alerts.pop()

        # Create report and save recording
        self.create_report(detection_data)


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
    
    def trigger_recording_save(self, detection_data):
        """Trigger your storage system to save the last few minutes."""
        st.app.session_state.app.recorder.save_replay()
        recording_data = {
            "timestamp": datetime.now(),
            "threat_type": detection_data["threat_type"],
            "confidence": detection_data["confidence"],
            "duration_minutes": 3,  # TODO: Adjust duration time of recording
            "video_path": f"storage/video_{int(time.time())}.mp4",
            "audio_path": f"storage/audio_{int(time.time())}.wav",
            "size_mb": random.randint(50, 200),
        }
        return recording_data


class Helpers:
    @staticmethod
    def start_detection_process():
        """Start the threat detection process"""
        if st.session_state.detection_running:
            return

        st.session_state.detection_running = True
        st.session_state.app.start()
        st.success("Threat detection process started")

    @staticmethod
    def stop_detection_process():
        """Stop the threat detection process"""
        if not st.session_state.detection_running:
            return

        st.session_state.detection_running = False
        st.session_state.app.stop()
        st.info("Threat detection process stopped")


class Renderer(HomeEdgeApp):
    def __init__(self, app):
        self.app = app

    def render_home_page(self):
        """Render the entire home page with header, navigation, and popup alerts"""
        self.render_header()
        self.render_navigation()

    def render_header(self):
        """Render the main header"""
        st.markdown(strings.MAIN_HEADER, unsafe_allow_html=True)

    def render_navigation(self):
        """Render navigation tabs"""
        tabs = st.tabs(["Control Dashboard", "Archives", "Settings"])

        with tabs[0]:
            st.session_state.current_page = "Control Dashboard"
            self.render_control_dashboard()

        with tabs[1]:
            st.session_state.current_page = "Archives"
            self.render_archives()

        with tabs[2]:
            st.session_state.current_page = "Settings"
            self.render_settings()

    # ------ CONTROL DASHBOARD PAGE ------

    def render_control_dashboard(self):
        """Render the control dashboard with detection controls, automatic schedule settings, recent alerts, and performance metrics"""

        self.render_detection_controls()
        st.divider()

        # Main Interface: Auto-Schedule + Detection Settings
        col_sch, col_sens = st.columns(2)
        with col_sch:
            self.render_auto_schedule_settings()
        with col_sens:
            self.render_detection_sensitivity_settings()
        st.divider()

        # 3. Recent Alerts (Kept)
        self.render_alerts_section()

        # 4. Performance Metrics
        self.render_performance_metrics()

    def render_detection_controls(self):
        """Render detection control buttons and status."""

        col1, col2, col3 = st.columns(3)

        # Start detection button
        with col1:
            if st.button(
                "**Start Detection**",
                disabled=st.session_state.detection_running,
                use_container_width=True,
                type="primary",
            ):
                Helpers.start_detection_process()

        # Stop detection button
        with col2:
            if st.button(
                "**Stop Detection**",
                disabled=not st.session_state.detection_running,
                use_container_width=True,
                type="secondary",
            ):
                Helpers.stop_detection_process()

        # On/off detection status
        with col3:
            status_class = (
                "detection-running"
                if st.session_state.detection_running
                else "detection-stopped"
            )
            status_text = "RUNNING" if st.session_state.detection_running else "STOPPED"
            st.markdown(
                strings.STATUS_CLASS(status_class, status_text), unsafe_allow_html=True
            )

    # TODO: Implement automatic start/stop logic based on schedule
    def render_auto_schedule_settings(self):
        """Renders the new automatic start/stop scheduling section."""

        st.subheader("Automatic Start/Stop Schedule")

        schedule = st.session_state.schedule_config
        new_enable = st.checkbox(
            "Enable Automatic Scheduling", value=schedule["enable"]
        )

        col1, col2 = st.columns(2)
        with col1:
            new_start_time = st.time_input(
                "Start Time (Detection ON)", value=schedule["start_time"]
            )
        with col2:
            new_stop_time = st.time_input(
                "Stop Time (Detection OFF)", value=schedule["stop_time"]
            )

        st.markdown("**Select Active Days:**")

        day_options = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        new_days = st.multiselect("Days to Run", day_options, default=schedule["days"])

        # Update logic
        new_schedule = {
            "enable": new_enable,
            "start_time": new_start_time,
            "stop_time": new_stop_time,
            "days": new_days,
        }

        if new_schedule != st.session_state.schedule_config:
            st.session_state.schedule_config = new_schedule
            st.success("Automatic schedule updated!")

        if new_enable:
            st.info(
                f"Scheduled to run from **{new_start_time.strftime('%H:%M')}** to **{new_stop_time.strftime('%H:%M')}** on selected days."
            )

    # TODO: Pass new sensitivity settings to backend detector
    def render_detection_sensitivity_settings(self):
        """Renders the core sensitivity controls in a clean format."""
        st.subheader("Detection Sensitivity Settings")

        config = st.session_state.detection_config

        person_sens = st.slider(
            "Person Detection Sensitivity",
            0,
            100,
            config["person_sensitivity"],
            key="person_sens_dash",
        )

        motion_sens = st.slider(
            "Motion Sensitivity",
            0,
            100,
            config["motion_sensitivity"],
            key="motion_sens_dash",
        )

        audio_sens = st.slider(
            "Audio Sensitivity",
            0,
            100,
            config["audio_sensitivity"],
            key="audio_sens_dash",
        )

        sound_thresh = st.slider(
            "Sound Threshold (Alert trigger)",
            0,
            100,
            config["sound_threshold"],
            key="sound_thresh_dash",
        )

        # Update configuration
        new_config = {
            "person_sensitivity": person_sens,
            "motion_sensitivity": motion_sens,
            "audio_sensitivity": audio_sens,
            "sound_threshold": sound_thresh,
        }

        if new_config != st.session_state.detection_config:
            st.session_state.detection_config = new_config
            st.toast("Sensitivity settings updated!")

    # TODO: Test with populated archives/alerts
    # TODO: Do these alerts automatically update?
    def render_alerts_section(self):
        """Render recent alerts section"""
        st.subheader("Recent Alerts")

        if not st.session_state.alerts:
            st.info("No alerts detected. System monitoring active.")
            return

        for alert in st.session_state.alerts[:5]:  # Show only top 5 alerts
            with st.container():
                st.markdown(strings.METRIC_CARD(alert), unsafe_allow_html=True)

        # Add a link to archives
        st.markdown(
            f"[View All {len(st.session_state.alerts)} Alerts in Archives Tab](#)",
            unsafe_allow_html=True,
        )

    # TODO: Do these performance metrics automatically update?
    def render_performance_metrics(self):
        """Renders the core performance metrics."""

        st.subheader("Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)
        metrics = st.session_state.performance_metrics

        with col1:
            st.metric("Frames Per Second (FPS)", metrics["fps"])
        with col2:
            st.metric("Inference Latency", f"{metrics['latency']}ms")
        with col3:
            st.metric("NPU/CPU Usage", f"{metrics['npu_usage']}%")
        with col4:
            st.metric("Average Confidence", f"{metrics['confidence']}%")

    # ------ ARCHIVES PAGE ------
    # TODO: Test archives page with reports
    def render_archives(self):
        """Render the threat detection archives section"""
        st.subheader("Threat Detection Archives")

        if not st.session_state.archived_reports:
            st.info(
                "No archived reports available. Threat detections will appear here."
            )
            return

        # Archive controls
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                ["Newest First", "Oldest First", "Highest Confidence", "Threat Type"],
            )

        with col2:
            filter_type = st.selectbox(
                "Filter by Type", ["All", "Person", "Motion", "Sound", "Anomaly"]
            )

        with col3:
            if st.button("Clear All Archives"):
                st.session_state.archived_reports = []
                st.success("Archives cleared")
                st.experimental_rerun()

        # Sort and filter reports
        reports = st.session_state.archived_reports.copy()
        if filter_type != "All":
            reports = [
                r for r in reports if r["threat_type"].lower() == filter_type.lower()
            ]
        if sort_by == "Oldest First":
            reports = sorted(reports, key=lambda x: x["timestamp"])
        elif sort_by == "Highest Confidence":
            reports = sorted(reports, key=lambda x: x["confidence"], reverse=True)
        elif sort_by == "Threat Type":
            reports = sorted(reports, key=lambda x: x["threat_type"])

        # Display reports
        st.write(f"Found {len(reports)} reports")

        for report in reports:
            with st.expander(
                f"Report #{report['id']}: {report['threat_type'].title()} - {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Threat Type:** {report['threat_type'].title()}")
                    st.write(f"**Confidence:** {report['confidence']:.1%}")
                    st.write(
                        f"**Timestamp:** {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    st.write(f"**Status:** {report['status']}")

                with col2:
                    st.write(
                        f"**Objects Detected:** {len(report.get('bounding_boxes', []))}"
                    )
                    if report.get("performance_metrics"):
                        metrics = report["performance_metrics"]
                        st.write(f"**FPS:** {metrics.get('fps', 'N/A')}")
                        st.write(f"**Latency:** {metrics.get('latency', 'N/A')}ms")
                        st.write(f"**NPU Usage:** {metrics.get('npu_usage', 'N/A')}%")

                # Recording data section
                if report.get("recording_data"):
                    st.write("**Recorded Media:**")
                    recording = report["recording_data"]
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"Duration: {recording['duration_minutes']} minutes")
                        st.write(f"Size: {recording['size_mb']} MB")

                    with col2:
                        # In production, these would be actual file download buttons
                        st.button(f"Download Video", key=f"video_{report['id']}")
                        st.button(f"Download Audio", key=f"audio_{report['id']}")

                    with col3:
                        st.button(f"Play Recording", key=f"play_{report['id']}")
                        st.button(f"Delete Recording", key=f"delete_{report['id']}")

                # Detection details
                if report.get("bounding_boxes"):
                    st.write("**Detection Details:**")
                    for i, box in enumerate(report["bounding_boxes"]):
                        st.write(
                            f"- Object {i+1}: {box.get('label', 'Unknown')} (confidence: {box.get('confidence', 0):.2f})"
                        )

    # ------ SETTINGS PAGE ------
    # TODO: Apply these settings to the backend
    def render_settings(self):
        """Render the settings section (Removed redundant detection config, kept storage/system info)"""
        st.subheader("System Configuration")

        # Storage Settings (Kept)
        st.write("**Storage Configuration**")

        col1, col2 = st.columns(2)

        with col1:
            buffer_duration = st.slider(
                "Buffer Duration (minutes)", 1, 10, 3, key="s_buff"
            )
            max_archive_size = st.slider(
                "Max Archive Size (GB)", 1, 50, 10, key="s_max"
            )

        with col2:
            auto_delete_days = st.slider(
                "Auto-delete Archives (days)", 7, 365, 30, key="s_del"
            )
            recording_quality = st.selectbox(
                "Recording Quality", ["Low", "Medium", "High", "Ultra"], key="s_qual"
            )

        if st.button("Save Storage Settings"):
            st.success("Storage settings saved")

        st.divider()

        # System Information
        st.write("**System Information**")
        cols = st.columns(1)
        with cols[0]:
            st.write("**System Status:**")
            st.write(f"- Detection Running: {st.session_state.detection_running}")
            st.write(f"- Total Alerts: {len(st.session_state.alerts)}")
            st.write(f"- Archived Reports: {len(st.session_state.archived_reports)}")
            st.write(f"- Current Threat Level: {st.session_state.threat_level:.1f}%")

    # ------ POPUP ALERTS ------
    # TODO: Connect this w/ actual detection alerts
    def render_popup_alert(self, alert):
        """Render popup alert for threats"""
        # Create popup using Streamlit's modal-like container
        alert = st.session_state.popup_alert_data
        with st.container():
            st.markdown(strings.THREAT_ALERT(alert), unsafe_allow_html=True)

        # TODO: Include actual sound alert + notification logic + user interaction to dismiss popup

        # Auto-dismiss logic (Simplified)
        if "alert_start_time" not in st.session_state:
            st.session_state.alert_start_time = time.time()
        if time.time() - st.session_state.alert_start_time > 10:
            st.session_state.show_popup_alert = False
            st.session_state.popup_alert_data = {}
            if "alert_start_time" in st.session_state:
                del st.session_state.alert_start_time


# Initialize and run the Streamlit app
if __name__ == "__main__":
    app = HomeEdgeApp()
    renderer = Renderer(app)
    renderer.render_home_page()
