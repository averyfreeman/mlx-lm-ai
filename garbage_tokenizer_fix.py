import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer
from mlx_lm import load
from mlx_lm.server import APIHandler

MODEL_PATH = "Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bit"

# 1. Setup your components safely
custom_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
model, _ = load(MODEL_PATH)

# 2. Let MLX's APIHandler mount its own internal FastAPI app automatically
handler = APIHandler(model, custom_tokenizer)
app = handler.app  # This exposes all standard OpenAI endpoints natively

if __name__ == "__main__":
    print("Starting native mlx_lm endpoint wrappers on http://127.0.0.1:8080...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
