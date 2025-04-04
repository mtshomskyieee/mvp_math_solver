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

# Create necessary files
if not os.path.isfile('new_tools.csv'):
    open('new_tools.csv', 'w').close()
if not os.path.isfile('math_problems_vector_store.faiss.index'):
    open('math_problems_vector_store.faiss.index', 'w').close()
if not os.path.isfile('math_problems_vector_store.faiss.mappings'):
    open('math_problems_vector_store.faiss.mappings', 'w').close()

if __name__ == "__main__":
    try:
        # Run the Streamlit app
        app()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
