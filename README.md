## Quickstart

### 1. Install dependencies
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Train tokenizer
```bash
python tokenizer/train_tokenizer.py
```

### 3. Prepare data
```bash
python data/prepare.py
```

### 4. Train model
```bash
# CPU
python train.py

# GPU (recommended)
python train.py --device cuda
```

### 5. Export + quantize
```bash
python export/export_gguf.py --checkpoint checkpoints/ckpt_0000500.pt
python export/quantize.py --weights export/model_weights.pt
```

### 6. Run server
```bash
python serve.py
# Open http://localhost:3001
```

### 7. Docker deployment
```bash
docker compose up --build
```

## Training Results
| Steps | Loss | Perplexity |
|-------|------|------------|
| 500   | 5.81 | 335        |

## Tech Stack
- PyTorch
- SentencePiece
- FastAPI + Uvicorn
- HuggingFace Datasets
- Docker
EOF