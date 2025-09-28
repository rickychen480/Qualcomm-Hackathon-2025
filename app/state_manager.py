import streamlit as st
import random
from datetime import datetime
from backend import HomeEdgeBackend

def initialize_session_state():
    """Initialize Streamlit session state variables"""

    # Storage settings
    if "storage_settings" not in st.session_state:
        st.session_state.storage_settings = {
            "buffer_duration": 0.1,
            # "buffer_duration": 3, # TODO: REINSTATE
            "max_archive_size": 10,
            "auto_delete_days": 30,
            "recording_quality": "Medium",
        }

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
    
    # Backend app instance
    if "app" not in st.session_state:
        st.session_state.app = HomeEdgeBackend(st.session_state)