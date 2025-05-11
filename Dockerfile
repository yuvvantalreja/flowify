FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for controlling cache locations
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_HOME=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/torch/sentence_transformers
ENV TORCH_HOME=/app/.cache/torch

# Create cache directories with proper permissions
RUN mkdir -p /app/.cache/huggingface/transformers
RUN mkdir -p /app/.cache/huggingface/datasets
RUN mkdir -p /app/.cache/torch/sentence_transformers
RUN mkdir -p /app/.cache/torch/hub

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow (CPU version to save space)
RUN pip install --no-cache-dir tensorflow-cpu==2.12.0

# Install additional dependencies directly
RUN pip install --no-cache-dir sentence-transformers>=2.2.2

# Download sentence-transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Run NLTK downloads
RUN python -m nltk.downloader punkt wordnet

# Copy the rest of the application
COPY . .

# Expose the port Hugging Face Spaces expects (7860)
EXPOSE 7860

# Run using uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"] 