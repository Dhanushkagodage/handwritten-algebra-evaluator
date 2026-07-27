"""
Persistent inference server for feedback-service, deployed as a Hugging
Face Space (Docker SDK) — see DEPLOY.md for the one-time setup.

Loads the public base model plus the private LoRA adapter (pulled from the
Hub with HF_TOKEN) once at startup, then serves it over a single
/generate endpoint matching what feedback_generator.py's
_run_space_inference() expects. A shared-secret X-API-Key header keeps the
endpoint from being called by anyone who stumbles onto the Space's public
URL, since the Space itself can't be private and still take plain HTTP
calls.
"""

import os

import torch
from fastapi import FastAPI, Header, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="feedback-service inference (HF Space)")

_base_model = os.environ["BASE_MODEL"]
_adapter_repo_id = os.environ["ADAPTER_REPO_ID"]
_hf_token = os.environ["HF_TOKEN"]
_api_key = os.environ["API_KEY"]

tokenizer = AutoTokenizer.from_pretrained(_base_model)
model = AutoModelForCausalLM.from_pretrained(_base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, _adapter_repo_id, token=_hf_token)
model.eval()
print(f"[hf_space] Loaded {_base_model} + adapter {_adapter_repo_id} on CPU (bfloat16)")


class GenerateRequest(BaseModel):
    messages: list[dict]


def _check_api_key(x_api_key: str = Header(default=None)) -> None:
    if x_api_key != _api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.post("/generate")
def generate(req: GenerateRequest, x_api_key: str = Header(default=None)):
    _check_api_key(x_api_key)

    prompt = tokenizer.apply_chat_template(
        req.messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
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
    return {"status": "ok", "base_model": _base_model, "adapter_repo_id": _adapter_repo_id}
