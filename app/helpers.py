import streamlit as st

def start_detection_process():
    """Start the threat detection process."""
    if st.session_state.detection_running:
        return

    st.session_state.detection_running = True
    st.session_state.app.start()
    st.success("Threat detection process started")

def stop_detection_process():
    """Stop the threat detection process."""
    if not st.session_state.detection_running:
        return
    
    st.session_state.detection_running = False
    st.session_state.app.stop()
    st.info("Threat detection process stopped")