import streamlit as st
import os
from backend import HomeEdgeApp

if "app" not in st.session_state:
    st.session_state.app = HomeEdgeApp()
    st.session_state.app.start()

st.title("HomeEdge Surveillance System")

if st.button("Simulate Threat"):
    detection = {"threat_type": "intruder"}
    report = st.session_state.app.handle_ml_detection_result(detection)
    
    st.success("Threat detected and report generated!")

    # Display report details
    st.subheader("Threat Report")
    st.write(f"**Type:** {report['threat_type']}")
    st.write(f"**Time:** {report['timestamp']}")
    st.write(f"**Severity:** {report['severity']}")
    st.write(f"**Description:** {report['description']}")
    st.write("**Actions Taken:**")
    for action in report['actions_taken']:
        st.write(f"- {action}")

    # Show video if exists
    video_path = report.get("video_path")
    if video_path and os.path.exists(video_path):
        st.video(video_path)
    else:
        st.info("No video available.")

# kind of need actual audio/videos, but tried simulating a threat with a pop up