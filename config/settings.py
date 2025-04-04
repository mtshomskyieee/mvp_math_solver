
# math_solver/config/settings.py
import os

# API keys
override_key = ''
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
    "(5 mod 2)",
    "round 1.578 to 2 decimals",
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