import streamlit as st
import os
from backend import HomeEdgeApp

# Initialize app backend once
if "app" not in st.session_state:
    st.session_state.app = HomeEdgeApp()
    st.session_state.app.start()

st.title("HomeEdge Surveillance System")

if st.button("Simulate Threat"):
    detection = {"threat_type": "intruder"}
    result = st.session_state.app.handle_ml_detection_result(detection)
    st.success("Saved replay!")

    video_path = result.get("video_path")
    if video_path and os.path.exists(video_path):
        st.video(video_path)
