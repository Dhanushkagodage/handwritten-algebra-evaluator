# Feedback Service — Module 03: Stepwise Feedback Generation

Generates step-by-step feedback (what's correct, what's missing, why marks
were reduced, how to improve) for a student's algebra answer, using a
LoRA-fine-tuned `Qwen2.5-3B-Instruct` model.

The dev machine has no local GPU, so training runs on a free **Google Colab /
Kaggle T4 GPU**. The trained LoRA adapter is pushed to a private repo on
**Hugging Face Hub**, and served from an always-on **Hugging Face Space**
(Gradio SDK + ZeroGPU) — the deployed service just calls that Space
through `gradio_client`, it never downloads or runs the model itself.
Unlike a live Colab session, the Space stays up permanently, so there's no
tunnel URL to re-copy every time you want to use it.

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
SPACE_ID=DhanushkaGodage/feedback-service-inference
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

## Part C — Run the 10 sample cases (Colab)

`tests/test_feedback.py` above exercises a single question. For broader
coverage there are ten more sample payloads in `tests/sample_inputs/`, each
with its own runner script, so you can step through them one at a time:

| Script | Method | Steps | Marks | What it covers |
|---|---|---|---|---|
| `test_case_01.py` | linear equation with brackets | 3 | 3.0/3.0 | full marks — no deduction text expected |
| `test_case_02.py` | quadratic formula | 4 | 2.5/4.0 | the standard mixed case |
| `test_case_03.py` | elimination | 4 | 3.0/4.0 | sign error only at the last step |
| `test_case_04.py` | completing the square | 4 | 2.0/4.0 | `partial` validity (dropped the ± root) |
| `test_case_05.py` | linear inequality | 3 | 2.0/3.0 | not reversing `<` on ÷ by a negative |
| `test_case_06.py` | simplifying algebraic fractions | 3 | 0.0/3.0 | zero-marks edge case |
| `test_case_07.py` | difference of squares | 2 | 1.5/2.0 | shortest answer |
| `test_case_08.py` | index laws | 5 | 3.0/5.0 | longest chain |
| `test_case_09.py` | substitution | 4 | 2.0/4.0 | optional `error_description` / scheme `description` |
| `test_case_10.py` | remainder theorem | 4 | 3.0/4.0 | Module 02's legacy `is_correct` bool |

### Two backends

| `--backend` | How the model runs | Needs |
|---|---|---|
| `space` (default) | through the Hugging Face Space over `gradio_client` — the same path the deployed service uses | `SPACE_ID`, `API_KEY`, `HF_TOKEN`; costs ZeroGPU quota; no GPU needed locally |
| `local` | base model + LoRA adapter loaded straight onto the Colab/Kaggle **T4** | a GPU runtime, `ADAPTER_PATH`, `HF_TOKEN`; costs no ZeroGPU quota |

Both share the same prompt builder, parser and validation, so the printed
`FeedbackResponse` has the same structure either way — only the wording
differs between runs (the model samples at `temperature=0.7`).

### Cell 1 — get the code and the GPU dependencies

Runtime → Change runtime type → **T4 GPU** → Save.

```python
import os
%cd /content
if not os.path.exists("handwritten-algebra-evaluator"):
    !git clone -b feedback-service-evalution-metices https://github.com/Dhanushkagodage/handwritten-algebra-evaluator.git
%cd /content/handwritten-algebra-evaluator/services/feedback-service
!pip install -r requirements-train.txt
!pip install gradio_client python-dotenv
```

### Cell 2 — pick the backend once for the whole session

```python
# (a) Colab T4 + your LoRA adapter from the Hub — free, no ZeroGPU quota
os.environ["BACKEND"]      = "local"
os.environ["ADAPTER_PATH"] = "DhanushkaGodage/qwen25-feedback-lora"
os.environ["HF_TOKEN"]     = "hf_..."

# (b) or the ZeroGPU Space API instead — comment out (a) and use this
# os.environ["BACKEND"]  = "space"
# os.environ["SPACE_ID"] = "DhanushkaGodage/feedback-service-inference"
# os.environ["API_KEY"]  = "<the Space's API_KEY secret>"
# os.environ["HF_TOKEN"] = "hf_..."
```

If you just trained in this same session, the adapter is already on disk —
use `os.environ["ADAPTER_PATH"] = "./lora-adapter"` instead of the Hub id.

### Cells 3-12 — one case per cell

```python
!python tests/test_case_01.py
```
```python
!python tests/test_case_02.py
```
…through `test_case_10.py`. Each prints the question, the full generated
`FeedbackResponse` as JSON, and a pass/fail line.

To override the backend for a single run without changing Cell 2:

```python
!python tests/test_case_03.py --backend space
```

### Optional — all ten in one go

On the `local` backend each separate script reloads Qwen2.5-3B from scratch
(several minutes each). This loads it once and runs all ten:

```python
!python tests/run_all_cases.py
!python tests/run_all_cases.py --cases 2,5,9   # or just a subset
```

Avoid `run_all_cases.py --backend space` — ten Space calls back-to-back will
eat a noticeable chunk of your ZeroGPU quota.

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
py -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('SPACE_ID'))"
```

---

## Troubleshooting

- **`invalid or missing api_key` calling the Space** — `API_KEY` in `.env`
  doesn't match the Space's `API_KEY` secret. Check both for stray
  whitespace.
- **`ModuleNotFoundError: No module named 'app'`** — the Colab cell ran from
  the wrong directory. Re-run the `%cd /content/handwritten-algebra-evaluator/services/feedback-service`
  line before the failing command.
- **`ls: cannot access 'lora-adapter'`** or the whole repo folder is missing
  from `/content` — the Colab runtime fully reset (disk wiped). Redo Cells
  1–3 from scratch; training only takes ~2 minutes.
- **`ModuleNotFoundError: No module named 'dotenv'` locally** — run
  `py -m pip install -r requirements.txt` from this folder.
- **Space request is very slow** — the first request after the Space (or
  ZeroGPU allocation) has been idle takes longer; see
  `app/serving/hf_space/DEPLOY.md` for tuning notes. Consistently slow
  warm requests usually mean ZeroGPU wasn't selected under Settings →
  Hardware and it's falling back to CPU.
- **Space returns a stale-looking adapter after retraining** — Spaces load
  the model once at container startup, so a fresh `upload_folder()` push
  from Colab won't be picked up until the Space restarts (use "Restart
  this Space" in its settings, or just wait for it to sleep/wake on the
  free tier).

For more background on the compute setup, see
`app/training/COLAB.md`.
