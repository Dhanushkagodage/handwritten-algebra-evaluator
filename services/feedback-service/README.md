# Feedback Service — Module 03: Stepwise Feedback Generation

Generates step-by-step feedback (what's correct, what's missing, why marks
were reduced, how to improve) for a student's algebra answer, using a
LoRA-fine-tuned `Qwen2.5-3B-Instruct` model.

The dev machine has no local GPU, so training runs on a free **Google Colab /
Kaggle T4 GPU**. The trained LoRA adapter is pushed to a private repo on
**Hugging Face Hub**, and served from a free, always-on **Hugging Face
Space** — the deployed service just calls that Space's URL over HTTP, it
never downloads or runs the model itself. Unlike a live Colab session, the
Space stays up permanently, so there's no tunnel URL to re-copy every time
you want to use it.

---

## Golden rule

Run all of Part A's cells back-to-back without long pauses between them. If
any cell ever errors with `No such file or directory` or `No module named
'app'`, the Colab runtime lost its working directory (or fully reset) —
just re-run **Cell 1** again (safe, won't re-clone if the folder already
exists) before continuing.

---

## Part A — Train the model and publish it (Google Colab)

Open [colab.research.google.com](https://colab.research.google.com) → **New
notebook** → **Runtime → Change runtime type → T4 GPU → Save**.

Run each cell below in its own cell, in order.

### Cell 1 — Get the code + install training dependencies

```python
import os
%cd /content
if not os.path.exists("handwritten-algebra-evaluator"):
    !git clone -b move-feedback-finetune-modal-to-huggingface https://github.com/Dhanushkagodage/handwritten-algebra-evaluator.git
%cd /content/handwritten-algebra-evaluator/services/feedback-service
!pip install -r requirements-train.txt
```

### Cell 2 — Generate the training data

```python
%cd /content/handwritten-algebra-evaluator/services/feedback-service
!python -m app.training.dataset
```

Expect: `Prepared 58 examples → ...` and `Prepared 10 examples → ...`

### Cell 3 — Train

```python
import os
%cd /content/handwritten-algebra-evaluator/services/feedback-service
os.environ["OUTPUT_DIR"] = "./lora-adapter"
os.environ["WANDB_REPORT_TO"] = "none"
!python -m app.training.train
```

Takes roughly 2 minutes for the default ~58-example dataset. Expect the last
line: `LoRA adapter saved to: ./lora-adapter`

### Cell 4 — Push the trained adapter to Hugging Face Hub

```python
%cd /content/handwritten-algebra-evaluator/services/feedback-service
from huggingface_hub import login, HfApi

login(token="hf_...")  # write-scoped token — or os.environ["HF_TOKEN"]

api = HfApi()
REPO_ID = "DhanushkaGodage/qwen25-feedback-lora"
api.create_repo(REPO_ID, private=True, exist_ok=True)
api.upload_folder(folder_path="./lora-adapter", repo_id=REPO_ID)
```

Confirm it worked: `https://huggingface.co/<REPO_ID>/tree/main` should show
`adapter_model.safetensors` + `adapter_config.json`. Once this finishes, the
Colab session can be closed — the Hugging Face Space (set up once, see
`app/serving/hf_space/DEPLOY.md`) pulls whatever adapter is at `REPO_ID` and
stays running independently of Colab.

---

## Part B — Test it (your laptop)

### 1. Point `.env` at your Hugging Face Space

Open `.env` in this folder and set:

```
SPACE_API_URL=https://dhanushkagodage-feedback-service-inference.hf.space/generate
API_KEY=<the shared secret you set as the Space's API_KEY secret>
```

See `app/serving/hf_space/DEPLOY.md` for how to create the Space (one-time
setup) if you haven't already.

### 2. Install local dependencies (only needed once)

```powershell
cd services\feedback-service
py -m pip install -r requirements.txt
```

> Use `py`, not `python` — on Windows, plain `python` is often hijacked by a
> Microsoft Store install-stub that does nothing.

### 3. Run the test

```powershell
py tests\test_feedback.py
```

You should get printed JSON with `step_feedback`, `overall_feedback`, and
`improvement_suggestions` for the sample question in
`tests\sample_input.json`.

### 4. Test your own question

Edit `tests/sample_input.json` with your own `question_text`,
`student_steps`, and `marking_scheme` (see the schema in
`app/models/schemas.py`), then re-run:

```powershell
py tests\test_feedback.py
```

---

## Useful commands (Colab)

Handy one-liners for checking on things while training in Colab.

**Check you got a GPU:**
```
!nvidia-smi
```

**Check current working directory + confirm the repo/adapter are there:**
```
!pwd
!ls
!ls lora-adapter
```

**Save the trained adapter to Google Drive so it survives a runtime reset:**
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r lora-adapter /content/drive/MyDrive/lora-adapter-backup
```
To restore it later instead of retraining:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/lora-adapter-backup ./lora-adapter
```

---

## Useful commands (your laptop)

**Confirm which Python actually runs (`python` can be a dead Store alias on Windows):**
```powershell
py --version
```

**Reinstall dependencies from scratch:**
```powershell
py -m pip install -r requirements.txt --force-reinstall
```

**Run the test with more visible output / stop on first failure:**
```powershell
py -u tests\test_feedback.py
```

**Quickly check your `.env` is picked up correctly:**
```powershell
py -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('SPACE_API_URL'))"
```

---

## Troubleshooting

- **`401 Unauthorized` calling the Space** — `API_KEY` in `.env` doesn't
  match the Space's `API_KEY` secret. Check both, and make sure you didn't
  leave a trailing `/generate` or whitespace in the Space secret itself.
- **`ModuleNotFoundError: No module named 'app'`** — the Colab cell ran from
  the wrong directory. Re-run the `%cd /content/handwritten-algebra-evaluator/services/feedback-service`
  line before the failing command.
- **`ls: cannot access 'lora-adapter'`** or the whole repo folder is missing
  from `/content` — the Colab runtime fully reset (disk wiped). Redo Cells
  1–3 from scratch; training only takes ~2 minutes.
- **`ModuleNotFoundError: No module named 'dotenv'` locally** — run
  `py -m pip install -r requirements.txt` from this folder.
- **Space request is very slow (30-90s+)** — expected; the free Space runs
  on CPU only. See `app/serving/hf_space/DEPLOY.md` for tuning notes.
- **Space returns a stale-looking adapter after retraining** — Spaces load
  the model once at container startup, so a fresh `upload_folder()` push
  from Colab won't be picked up until the Space restarts (use "Restart
  this Space" in its settings, or just wait for it to sleep/wake on the
  free tier).

For more background on the compute setup, see
`app/training/COLAB.md`.
