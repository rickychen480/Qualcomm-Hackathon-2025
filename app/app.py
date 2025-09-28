import streamlit as st

import strings
from state_manager import initialize_session_state
from app_logic import HomeEdgeApp
from renderer import Renderer

# --- Page Configuration ---
st.set_page_config(
    page_title="HomeEdge Security Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(strings.CSS, unsafe_allow_html=True)

# --- Main Application Runner ---
def main():
    """Main function to run the Streamlit application."""
    # 1. Initialize session state variables.
    initialize_session_state()

    # 2. Instantiate the logic and renderer classes.
    app_logic = HomeEdgeApp()
    renderer = Renderer()

    # 3. Render the main page structure.
    renderer.render_home_page()

    # 4. Conditionally render the popup alert outside the main flow.
    if st.session_state.get("show_popup_alert"):
        renderer.render_popup_alert(st.session_state.popup_alert_data)

if __name__ == "__main__":
    main()