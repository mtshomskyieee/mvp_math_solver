# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.cargo/bin:${PATH}" \
    VIRTUAL_ENV="/app/venv"

# Install system dependencies for Rust and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for building rust_math_extensions)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Source cargo environment in the same RUN command
RUN . $HOME/.cargo/env && rustup default stable

# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# for rust compilation/integration
RUN pip install maturin
# for unit tests
RUN pip install pytest

# Copy the entire project
COPY agents /app/agents
COPY core /app/core
COPY config /app/config
COPY ui /app/ui
COPY utils /app/utils
COPY workflows /app/workflows
COPY main.py /app
COPY *.faiss* /app/
COPY *.csv /app/

# Build the Rust extensions
WORKDIR /app/core
RUN . $HOME/.cargo/env && maturin develop
WORKDIR /app

# Set port for Streamlit (default is 8501)
EXPOSE 8501

# Run the application using streamlit
CMD ["streamlit", "run", "main.py"]