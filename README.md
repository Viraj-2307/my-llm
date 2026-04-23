cat > README.md << 'EOF'
---
title: My LLM
emoji: 🧠
colorFrom: green
colorTo: black
sdk: docker
app_port: 7860
pinned: true
---

# My LLM — Built from Scratch

A complete, production-grade LLM built entirely from scratch.

## Architecture
- Decoder-only transformer (GPT-style)
- 35M parameters
- Grouped Query Attention (GQA)
- SwiGLU FFN + RoPE + RMSNorm
- Int8 quantization for CPU inference

## Quickstart
\`\`\`bash
docker compose up --build
# Open http://localhost:3001
\`\`\`

## Training Results
| Steps | Loss | Perplexity |
|-------|------|------------|
| 500   | 5.81 | 335        |

## Tech Stack
PyTorch · SentencePiece · FastAPI · Docker · HuggingFace
EOF