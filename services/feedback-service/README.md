# Feedback Service — Module 03: Stepwise Feedback Generation

Generates step-by-step feedback (what's correct, what's missing, why marks
were reduced, how to improve) for a student's algebra answer, using a
LoRA-fine-tuned `Qwen2.5-3B-Instruct` model.

The dev machine has no local GPU, so training runs on a free **Google Colab /
Kaggle T4 GPU**. The trained model is served from that same Colab session
over a temporary public URL (Cloudflare Quick Tunnel), and the deployed
service just calls that URL over HTTP — it never downloads or runs the model
itself.

---

## Golden rule

Run all of Part A's cells back-to-back without long pauses between them. If
any cell ever errors with `No such file or directory` or `No module named
'app'`, the Colab runtime lost its working directory (or fully reset) —
just re-run **Cell 1** again (safe, won't re-clone if the folder already
exists) before continuing.

---

## Part A — Train and serve the model (Google Colab)

Open [colab.research.google.com](https://colab.research.google.com) → **New
notebook** → **Runtime → Change runtime type → T4 GPU → Save**.

Run each cell below in its own cell, in order.

### Cell 1 — Get the code + install training dependencies

```python
import os
%cd /content
if not os.path.exists("handwritten-algebra-evaluator"):
    !git clone -b feedback-generation-module https://github.com/Dhanushkagodage/handwritten-algebra-evaluator.git
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

### Cell 4 — Start the model server

```python
%cd /content/handwritten-algebra-evaluator/services/feedback-service
!pip install fastapi uvicorn
!nohup uvicorn colab_server:app --host 0.0.0.0 --port 8000 --app-dir app/training > server.log 2>&1 &
```

### Cell 5 — Confirm the server actually came up before moving on

```python
import time
time.sleep(30)
!cat server.log
```

You must see a line like `Uvicorn running on http://0.0.0.0:8000` and
`Loaded ... + adapter ./lora-adapter`. **Do not proceed to Cell 6 until you
see this** — a traceback here means the server crashed (check for missing
packages, wrong `--app-dir`, or GPU memory errors).

### Cell 6 — Open the public tunnel

```python
%cd /content/handwritten-algebra-evaluator/services/feedback-service
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!./cloudflared-linux-amd64 tunnel --url http://localhost:8000
```

This cell runs forever — that's expected, leave it running. Copy the printed
URL, e.g. `https://random-words.trycloudflare.com`.

**Limitation:** this URL is temporary. It changes every time the tunnel or
the Colab session restarts, and stops working once Colab disconnects (idle
timeout, or the free-tier multi-hour cap). Start Colab before you need it,
copy the fresh URL, use it, and it's fine if it goes offline afterward.

---

## Part B — Test it (your laptop)

### 1. Set the tunnel URL in `.env`

Open `.env` in this folder and set (note the `/generate` suffix):

```
TUNNEL_API_URL=https://random-words.trycloudflare.com/generate
```

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

Handy one-liners for checking on things while training/serving in Colab.

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

**Check if the server process is running:**
```
!ps aux | grep uvicorn
```

**Tail the server logs without waiting:**
```
!cat server.log
```

**Check the server directly from inside Colab (bypasses the tunnel):**
```
!curl http://localhost:8000/health
```

**Kill the server (e.g. to restart it after a code change):**
```
!pkill -f uvicorn
```
Then re-run Cell 4 to start it again.

**Send a manual test request from inside Colab:**
```
!curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hello in one word."}]}'
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
py -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('TUNNEL_API_URL'))"
```

---

## Troubleshooting

- **`502 Bad Gateway` when calling the tunnel URL** — the tunnel itself is
  fine, but nothing is listening on port 8000 inside Colab. Re-check Cell 5's
  `server.log` for a crash, and confirm `!pwd` inside Colab shows
  `.../services/feedback-service` (a wrong/reset working directory is the
  most common cause).
- **`ModuleNotFoundError: No module named 'app'`** — the Colab cell ran from
  the wrong directory. Re-run the `%cd /content/handwritten-algebra-evaluator/services/feedback-service`
  line before the failing command.
- **`ls: cannot access 'lora-adapter'`** or the whole repo folder is missing
  from `/content` — the Colab runtime fully reset (disk wiped). Redo Cells
  1–4 from scratch; training only takes ~2 minutes.
- **`ModuleNotFoundError: No module named 'dotenv'` locally** — run
  `py -m pip install -r requirements.txt` from this folder.
- **Tunnel URL stopped working after a while** — Colab sessions disconnect
  after idle time or a few hours on the free tier. Redo Cells 4–6 to get a
  fresh URL and update `.env`.

For more background on the compute setup, see
`app/training/COLAB.md`.
