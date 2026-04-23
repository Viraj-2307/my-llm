FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints export data/shards tokenizer \
             finetune/checkpoints frontend

# HuggingFace requires port 7860
EXPOSE 7860

# HuggingFace runs as non-root user 1000
RUN useradd -m -u 1000 user
USER user

CMD ["python", "serve.py"]