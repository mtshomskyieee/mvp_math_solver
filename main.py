import os
import streamlit as st
from config.settings import OPENAI_API_KEY
from ui.main_app import app
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger("main")

# Ensure API key is set
override_key = "your key here"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', override_key)


if __name__ == "__main__":
    try:
        # Run the Streamlit app
        app()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
