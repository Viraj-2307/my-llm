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
RUN mkdir -p checkpoints export data/shards tokenizer finetune/checkpoints

EXPOSE 3001

CMD ["python", "serve.py"]