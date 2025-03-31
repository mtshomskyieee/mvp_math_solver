# math_solver/utils/logging_utils.py
import logging


def setup_logger(name=None):
    """Configure and return a logger instance."""
    logger = logging.getLogger(name or __name__)

    # Configure logging if not already configured
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    return logger