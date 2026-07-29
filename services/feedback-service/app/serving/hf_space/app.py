"""
Persistent inference server for feedback-service, deployed as a Hugging
Face Space (Gradio SDK, ZeroGPU) — see DEPLOY.md for the one-time setup.

Loads the public base model plus the private LoRA adapter (pulled from the
Hub with HF_TOKEN) once on the first request — must happen inside the
@spaces.GPU call, not at import time, since ZeroGPU only attaches a real
GPU during that window — then serves it over a single "generate" API
endpoint matching what feedback_generator.py's _run_space_inference()
expects. gradio_client doesn't forward custom HTTP headers, so the same
shared-secret check the old FastAPI version did via X-API-Key is now a
second function argument instead — still keeps the endpoint from being
called by anyone who stumbles onto the Space's public URL, since the Space
itself can't be private and still take plain gradio_client calls without
each caller needing its own HF token.
"""

import json
import os

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_base_model = os.environ["BASE_MODEL"]
_adapter_repo_id = os.environ["ADAPTER_REPO_ID"]
_hf_token = os.environ["HF_TOKEN"]
_api_key = os.environ["API_KEY"]

tokenizer = AutoTokenizer.from_pretrained(_base_model)

_model = None


def _get_model():
    # ZeroGPU only attaches a real GPU inside a @spaces.GPU call, so the
    # adapter merge and .to("cuda") must happen here (on first request),
    # not at module import time — doing it at import time makes torch
    # think a GPU exists (spaces patches torch.cuda.is_available()) with
    # no physical device actually attached yet, which crashes with
    # "No CUDA GPUs are available".
    global _model
    if _model is None:
        model = AutoModelForCausalLM.from_pretrained(_base_model, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, _adapter_repo_id, token=_hf_token)
        model = model.to("cuda")
        model.eval()
        _model = model
        print(f"[hf_space] Loaded {_base_model} + adapter {_adapter_repo_id} on ZeroGPU (bfloat16)")
    return _model


@spaces.GPU(duration=120)
def generate(messages_json: str, api_key: str) -> str:
    if api_key != _api_key:
        raise gr.Error("invalid or missing api_key", print_exception=False)

    model = _get_model()
    messages = json.loads(messages_json)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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
    return text.strip()


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="messages_json"),
        gr.Textbox(label="api_key"),
    ],
    outputs=gr.Textbox(label="text"),
    title="feedback-service inference (HF Space)",
    description="Not meant to be browsed directly — see DEPLOY.md in the main repo.",
    api_name="generate",
)

if __name__ == "__main__":
    demo.queue().launch()
