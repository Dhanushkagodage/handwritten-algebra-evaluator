# Training on Google Colab / Kaggle (free T4, 16GB VRAM)

The user's dev machine has no local GPU, so LoRA fine-tuning of
`Qwen2.5-3B-Instruct` runs on a free Colab (or Kaggle) GPU session instead.
Everything below runs in Colab/Kaggle cells (`!` prefix = shell command,
otherwise plain Python in the same cell/notebook).

## 1. Set up the runtime

1. Colab: **Runtime → Change runtime type → T4 GPU**. (Kaggle: enable a GPU
   accelerator in the notebook settings.)

## 2. Get the code and install dependencies

```python
import os
%cd /content
if not os.path.exists("handwritten-algebra-evaluator"):
    !git clone -b move-feedback-finetune-modal-to-huggingface https://github.com/Dhanushkagodage/handwritten-algebra-evaluator.git
%cd handwritten-algebra-evaluator/services/feedback-service
```
```
!pip install -r requirements-train.txt
```
Safe to re-run this cell any time — it won't re-clone (avoiding a nested,
doubled-up directory) and it always pulls the
`move-feedback-finetune-modal-to-huggingface` branch, not whatever `main`
currently contains.

### If you need to delete and re-clone from scratch

Always `%cd /content` **before** deleting the repo folder. Deleting the
folder you're currently standing in breaks every shell command (`!ls`,
`!pip`, etc.) until you `%cd` somewhere that still exists:

```python
%cd /content
!rm -rf /content/handwritten-algebra-evaluator
```

Then re-run the clone cell above as normal.

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

## 5. Push the trained adapter to Hugging Face Hub

The deployed feedback-service never downloads or trains the model itself —
it calls a persistent Hugging Face Space instead (see
`app/serving/hf_space/DEPLOY.md` for the one-time Space setup). This Colab
session's only job after training is to publish the adapter so the Space
can pull it:

```python
from huggingface_hub import login, HfApi

login(token="hf_...")  # write-scoped token — or os.environ["HF_TOKEN"]

api = HfApi()
REPO_ID = "DhanushkaGodage/qwen25-feedback-lora"
api.create_repo(REPO_ID, private=True, exist_ok=True)
api.upload_folder(folder_path="./lora-adapter", repo_id=REPO_ID)
```

Confirm it worked by checking
`https://huggingface.co/<REPO_ID>/tree/main` shows `adapter_model.safetensors`
and `adapter_config.json`. Once pushed, this Colab session can be closed —
the Space (which stays up independently) always serves the latest adapter
you push here, and every future re-train just needs this same push step to
update it live.

## Kaggle notes

Kaggle uses the exact same commands from step 2 onward — the only difference
is how the repo gets onto the machine (Kaggle's internet-enabled kernel +
`!git clone`, same as Colab, or "Add Data" if you'd rather upload a zip).
