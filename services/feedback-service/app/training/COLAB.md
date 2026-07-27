# Training on Google Colab / Kaggle (free T4, 16GB VRAM)

The user's dev machine has no local GPU, so LoRA fine-tuning of
`Qwen2.5-3B-Instruct` runs on a free Colab (or Kaggle) GPU session instead.
Everything below runs in Colab/Kaggle cells (`!` prefix = shell command,
otherwise plain Python in the same cell/notebook).

## 1. Set up the runtime

1. Colab: **Runtime → Change runtime type → T4 GPU**. (Kaggle: enable a GPU
   accelerator in the notebook settings.)

## 2. Get the code and install dependencies

```
!git clone <repo-url>
%cd handwritten-algebra-evaluator/services/feedback-service
!pip install -r requirements-train.txt
```

## 3. Regenerate the dataset

```
!python -m app.training.dataset
```
This writes `app/training/data/raw_annotations.json`,
`feedback_dataset_train.json` (~58 examples), and `feedback_dataset_eval.json`
(~10 examples).

## 4. Train

```python
import os
os.environ["LOCAL_BASE_MODEL"] = "Qwen/Qwen2.5-3B-Instruct"
os.environ["OUTPUT_DIR"] = "./lora-adapter"
os.environ["WANDB_REPORT_TO"] = "none"   # or call wandb.login() first to keep W&B logging
```
```
!python -m app.training.train
```
Training runs 3 epochs over ~58 examples — typically 20-40 minutes on a T4.
Watch the logged `loss` (and `eval_loss`, since the eval split exists) —
both should trend down. When it finishes, `./lora-adapter/` contains
`adapter_model.safetensors` + `adapter_config.json`.

## 5. Serve the trained adapter and call it from the app

The deployed feedback-service never downloads or runs the model itself — it
always calls your trained model over HTTP. Serve it straight from this Colab
session:

```
!pip install fastapi uvicorn
!nohup uvicorn colab_server:app --host 0.0.0.0 --port 8000 --app-dir app/training &
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!./cloudflared-linux-amd64 tunnel --url http://localhost:8000
```
The last command prints a temporary public URL like
`https://random-words.trycloudflare.com`. Append `/generate` and put it in
`.env` locally as `TUNNEL_API_URL`, restart the service — generation happens
on Colab's GPU, nothing downloaded onto the machine running the service.

**Limitation:** this URL is temporary. It changes every time the tunnel or
the Colab session restarts, and stops working entirely once Colab
disconnects (idle timeout, or the free-tier multi-hour cap). Treat this as
"start Colab before a demo, copy the fresh URL into `.env`, present, then
it's fine if it goes offline afterward" — not something to leave configured
unattended.

## Kaggle notes

Kaggle uses the exact same commands from step 2 onward — the only difference
is how the repo gets onto the machine (Kaggle's internet-enabled kernel +
`!git clone`, same as Colab, or "Add Data" if you'd rather upload a zip).
