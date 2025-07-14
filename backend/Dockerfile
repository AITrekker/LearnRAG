# Use the latest NVIDIA PyTorch container for GPU compatibility
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set up the working directory
WORKDIR /app

# Upgrade pip and install essential system packages
RUN pip install --upgrade pip && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements-base.txt .
COPY requirements.txt .
COPY constraints.txt .

# Install PyTorch with specific CUDA version (REQUIRED for RTX 5070)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install base requirements (heavyweight packages)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade-strategy only-if-needed -c constraints.txt -r requirements-base.txt

# Install main requirements (lightweight packages)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade-strategy only-if-needed -c constraints.txt -r requirements.txt

# Test that ML libraries can be imported successfully
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    import transformers; print(f'Transformers version: {transformers.__version__}'); \
    print('Core ML libraries imported successfully!')"

# Copy the application source code
COPY app/ ./app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/cache \
    /app/models_cache /app/internal_files \
    /root/.cache/huggingface \
    /root/.cache/transformers

# Set environment variables for Python and caching
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/transformers
ENV MODELS_CACHE_DIR=/app/models_cache

# Expose the application port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--access-log", "--log-level", "info"]