"""
Local-GPU inference backend for the sample test cases.

The deployed service always calls the Hugging Face Space (ZeroGPU) through
gradio_client — see app/agents/feedback_generator.py. That path costs
ZeroGPU quota and needs the Space to be awake, which is inconvenient when
you just want to push a batch of sample inputs through the model.

This module offers the alternative used by tests/evaluation_suite/evaluate.py:
load the base model plus the LoRA adapter straight onto the Colab/Kaggle T4
and run generation in-process. No quota, no Space dependency.

It subclasses FeedbackGenerator and overrides *only* the two model-touching
methods, so prompt construction, response parsing, per-step validation and
the overall-feedback aggregation are the exact same code both backends run.
That is what makes a `local` result comparable to a `space` one.

Not importable on a machine without CUDA — torch/peft/transformers are not
in the service's requirements.txt (deliberately: the deployed service never
runs the model itself), so they are imported lazily inside load_model().
"""
import os
from typing import Dict, List, Optional, Tuple

from app.agents.feedback_generator import FeedbackGenerator

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER_PATH = "./lora-adapter"

# Loading Qwen2.5-3B takes several minutes on a Colab runtime, so cache it
# per (base_model, adapter_path) at module scope: run_all_cases.py builds a
# generator per case but only pays that cost once.
_MODEL_CACHE: Dict[Tuple[str, str], tuple] = {}


class LocalFeedbackGenerator(FeedbackGenerator):
    """FeedbackGenerator that runs the model here instead of on the Space."""

    def __init__(self, base_model: Optional[str] = None, adapter_path: Optional[str] = None):
        super().__init__()
        self._base_model = base_model or os.getenv("BASE_MODEL", DEFAULT_BASE_MODEL)
        self._adapter_path = adapter_path or os.getenv("ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
        self._model = None
        self._tokenizer = None

    async def load_model(self) -> None:
        if self._loaded:
            return
        self._model, self._tokenizer = _load_model(self._base_model, self._adapter_path)
        self._loaded = True

    async def _run_inference(self, messages: List[Dict]) -> str:
        import torch

        if not self._loaded:
            await self.load_model()

        # The Space applies the chat template server-side (hf_space/app.py);
        # doing it here keeps the token sequence the model sees identical.
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                # Same sampling settings as the Space, so the two backends
                # are compared like for like. (evaluate.py decodes greedily
                # instead — its metrics have to be reproducible.)
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return text.strip()


def _load_model(base_model: str, adapter_path: str):
    key = (base_model, adapter_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device — the 'local' backend needs a Colab/Kaggle GPU runtime. "
            "Use --backend space to call the Hugging Face Space instead."
        )

    # T4 (Turing) has no native bfloat16; fp16 is the correct choice there.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[local] loading {base_model} + adapter {adapter_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_path, token=os.getenv("HF_TOKEN"))
    model = model.to("cuda")
    model.eval()
    print(f"[local] ready ({str(dtype).split('.')[-1]})")

    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer
