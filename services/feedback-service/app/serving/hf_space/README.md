---
title: Feedback Service Inference
emoji: 🧮
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

Persistent inference server for the handwritten-algebra-evaluator
feedback-service (Module 03). Loads `Qwen2.5-3B-Instruct` plus a private
LoRA adapter from Hugging Face Hub and serves it over ZeroGPU via a
`generate` API endpoint. Not meant to be browsed directly — see
`DEPLOY.md` in the main repo for setup, and `../../../README.md` for how
the deployed service calls this Space.
