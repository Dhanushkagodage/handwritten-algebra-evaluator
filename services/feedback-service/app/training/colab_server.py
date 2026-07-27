"""
Standalone inference server that feedback-service calls over HTTP.

Run this INSIDE a Colab (or Kaggle) GPU notebook, not as part of the deployed
feedback-service. It loads the fine-tuned Qwen2.5-3B + LoRA adapter once and
serves it over a single /generate endpoint, matching what
`feedback_generator.py`'s `_run_tunnel_inference()` expects. Expose it to the
internet with a free Cloudflare Quick Tunnel — see app/training/COLAB.md.

Usage (inside Colab, after training or after loading a saved adapter):
    !pip install fastapi uvicorn
    !nohup uvicorn colab_server:app --host 0.0.0.0 --port 8000 &
"""

import os

import torch
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="feedback-service tunnel inference (Colab)")

_base_model = os.getenv("LOCAL_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
_adapter_dir = os.getenv("LORA_ADAPTER_DIR", "./lora-adapter")
_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(_base_model)
model = AutoModelForCausalLM.from_pretrained(
    _base_model,
    torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
).to(_device)
if os.path.isdir(_adapter_dir):
    model = PeftModel.from_pretrained(model, _adapter_dir)
    print(f"[colab_server] Loaded {_base_model} + adapter {_adapter_dir} on {_device}")
else:
    print(f"[colab_server] No adapter found at {_adapter_dir} — serving base {_base_model} un-tuned")


class GenerateRequest(BaseModel):
    messages: list[dict]


@app.post("/generate")
def generate(req: GenerateRequest):
    prompt = tokenizer.apply_chat_template(
        req.messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=600,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return {"text": text.strip()}


@app.get("/health")
def health():
    return {"status": "ok", "base_model": _base_model, "device": _device}
