"""
LoRA Fine-tuning Script for Qwen2.5-3B-Instruct
Module 03 — Stepwise Feedback Generation

Usage:
    python -m app.training.train
"""

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def train():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    dataset_path = os.getenv("DATASET_PATH", "app/training/data/feedback_dataset_train.json")
    eval_dataset_path = os.getenv("EVAL_DATASET_PATH", "app/training/data/feedback_dataset_eval.json")
    output_dir = os.getenv("OUTPUT_DIR", "./lora-adapter")
    run_name = os.getenv("WANDB_RUN_NAME", "qwen25-3b-feedback-lora")
    report_to = os.getenv("WANDB_REPORT_TO", "wandb")

    use_cuda = torch.cuda.is_available()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto",
    )
    model.enable_input_require_grads()

    # LoRA config — targets attention projection layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset — eval split is optional, falls back to train-only if missing
    has_eval = os.path.exists(eval_dataset_path)
    data_files = {"train": dataset_path}
    if has_eval:
        data_files["validation"] = eval_dataset_path
    dataset = load_dataset("json", data_files=data_files)

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=use_cuda,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if has_eval else "no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=report_to,
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_eval else None,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    train()
