"""
Production CPU inference server.
Usage: python serve.py
Then open: http://localhost:3001
"""
import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
import json
import time
import argparse
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from config import SMALL, InferenceConfig
from model import LLM

from finetune.lora import inject_lora, load_lora_weights

class QuantizedLinear(torch.nn.Module):
    """Drop-in replacement for nn.Linear using int8 weights."""
    def __init__(self, weight_int8, scale, bias=None):
        super().__init__()
        self.register_buffer("weight", weight_int8)
        self.register_buffer("scale", scale)
        self.bias = bias

    def forward(self, x):
        # Dequantize on the fly
        w = self.weight.float() * (self.scale.unsqueeze(-1) / 127.0)
        return F.linear(x, w, self.bias)


def load_quantized_model(weights_path: str, config) -> LLM:
    """Load int8 quantized weights into the model."""
    print(f"Loading quantized model from {weights_path}...")
    quantized_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Reconstruct float32 state dict for loading
    state_dict = {}
    for name, entry in quantized_dict.items():
        if entry["dtype"] == "int8":
            # Dequantize back to float32
            w = entry["data"].float() * (entry["scale"].unsqueeze(-1) / 127.0)
            state_dict[name] = w
        else:
            state_dict[name] = entry["data"]

    model = LLM(config)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded. Parameters: {model.num_params():,}")
    return model


# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="My LLM API",
    description="Production inference server for custom trained LLM",
    version="1.0.0",
)

if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model    = None
tokenizer = None
icfg     = InferenceConfig()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens:  Optional[int]  = 256
    temperature: Optional[float] = 0.8
    top_k:       Optional[int]  = 40
    top_p:       Optional[float] = 0.95


class GenerateResponse(BaseModel):
    text:           str
    tokens_generated: int
    time_seconds:   float
    tokens_per_sec: float

@app.on_event("startup")
async def startup():
    global model, tokenizer

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load("tokenizer/tokenizer.model")
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size()}")

    # Load base model
    model = load_quantized_model("export/model_int8.pt", SMALL)
    torch.set_num_threads(icfg.n_threads)

    # Load LoRA weights if provided
    lora_path = os.environ.get("LORA_PATH", "")
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        model = inject_lora(model, rank=8, alpha=16.0)
        model = load_lora_weights(model, lora_path)
        print("LoRA weights loaded successfully")
    else:
        print("No LoRA weights found, using base model")

    print(f"Server ready on http://localhost:3001")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model": "my-llm", "params": model.num_params()}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Encode prompt
    input_ids = [tokenizer.bos_id()] + tokenizer.encode(req.prompt, out_type=int)
    idx = torch.tensor([input_ids], dtype=torch.long)

    # Generate
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            idx,
            max_new_tokens = req.max_tokens,
            temperature    = req.temperature,
            top_k          = req.top_k,
            top_p          = req.top_p,
        )
    elapsed = time.time() - t0

    # Decode only the new tokens
    new_ids  = output_ids[0, len(input_ids):].tolist()
    new_text = tokenizer.decode(new_ids)
    new_text = new_text.replace("⁇", "").replace("<unk>", "").strip()

    return GenerateResponse(
        text             = new_text,
        tokens_generated = len(new_ids),
        time_seconds     = round(elapsed, 2),
        tokens_per_sec   = round(len(new_ids) / elapsed, 1),
    )


@app.post("/chat")
def chat(req: GenerateRequest):
    """Simple chat endpoint with prompt formatting."""
    formatted = f"User: {req.prompt}\nAssistant:"
    req.prompt = formatted
    return generate(req)

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=3001, reload=False)