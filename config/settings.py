
# math_solver/config/settings.py
import os

# API keys
override_key = "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"
OPENAI_API_KEY = os.getenv(
    'OPENAI_API_KEY',
    override_key
)

# Set API key in environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# LLM settings
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0

# Sample problems for UI
SAMPLE_PROBLEMS = [
    "What is the square root of -1",
    "What is 25 + 37?",
    "Multiply 13 by 7",
    "What is 144 divided by 12?",
    "What is the square root of 81?",
    "If x = 5 and y = 3, what is x^y?",
    "(2-3)*5^2 "
]

# Max retries for validation
MAX_VERIFICATION_RETRIES = 5