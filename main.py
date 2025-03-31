
import os
import streamlit as st
from config.settings import OPENAI_API_KEY
from ui.main_app import app
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger("main")

# Ensure API key is set
override_key = "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"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', override_key)

if __name__ == "__main__":
    try:
        # Configure Streamlit server settings
        st.set_page_config(page_title="Math Problem Solver")
        # Run the Streamlit app with proper server configuration
        app()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
