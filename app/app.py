import streamlit as st

import strings
from state_manager import initialize_session_state
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
    renderer = Renderer()

    # 3. Render the main page structure.
    renderer.render_home_page()

    # 4. Render the popup manager, which will handle showing alerts.
    renderer.render_popup_manager()

if __name__ == "__main__":
    main()