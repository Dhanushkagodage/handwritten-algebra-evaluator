# Deploying the inference Space (one-time setup)

This folder (`app.py`, `requirements.txt`, `README.md`) is the code for a
**Hugging Face Space** — a separate, always-on Gradio app hosted by
Hugging Face, independent of Colab. A Space is its own git repository at
`huggingface.co/spaces/<org>/<name>`, so this folder gets pushed there
directly; it isn't deployed via this GitHub repo.

You only need to do this once (or again if you want to change how the
server itself behaves — model swaps only need the Colab push step in
`app/training/COLAB.md`, not a redeploy here).

## 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Owner**: `DhanushkaGodage` (personal account) — so it's
   `huggingface.co/spaces/DhanushkaGodage/feedback-service-inference`.
3. **SDK**: **Gradio** → **Blank**.
4. **Visibility**: **Private** — the app also enforces its own `api_key`
   check on every call, so this is belt-and-suspenders, but it means
   callers need an HF read token in addition to the key (see step 4).
5. Click **Create Space**.
6. Once created, go to **Settings → Hardware** and select **ZeroGPU**. If
   your account doesn't offer it (some accounts need HF PRO), fall back to
   free **CPU basic** — `app.py` still runs there, just slower (see the
   git history of this file for the CPU/Docker version if you need to
   revert).

## 2. Set the Space's secrets

In the Space → **Settings → Variables and secrets**, add these as
**secrets** (not public variables):

| Name              | Value                                                              |
|-------------------|---------------------------------------------------------------------|
| `HF_TOKEN`        | A **read**-scoped token with access to your private adapter repo   |
| `ADAPTER_REPO_ID` | `DhanushkaGodage/qwen25-feedback-lora` (from the Colab push step)  |
| `BASE_MODEL`      | `Qwen/Qwen2.5-3B-Instruct`                                          |
| `API_KEY`         | Any random string you generate — shared secret for calling `generate` |

## 3. Push this folder to the Space

The Space repo is separate from this GitHub repo, so clone it alongside
and copy the files in:

```powershell
git clone https://huggingface.co/spaces/DhanushkaGodage/feedback-service-inference space-repo
cd space-repo
copy ..\handwritten-algebra-evaluator\services\feedback-service\app\serving\hf_space\* .
git add .
git commit -m "Deploy feedback-service inference server"
git push
```

You'll be prompted for Hugging Face credentials — use a token with write
access to the Space (Settings → Access Tokens on huggingface.co).

## 4. Confirm it's up

The Space builds and starts automatically (watch the **Logs** tab — first
boot downloads the ~6 GB base model, which can take several minutes). Once
you see
`Loaded Qwen/Qwen2.5-3B-Instruct + adapter DhanushkaGodage/... on ZeroGPU (bfloat16)`
in the logs and the Space shows **Running**, test it from your laptop
(needs `pip install gradio_client` — it's already in this service's
`requirements.txt`). Because the Space is private, you also need an HF
**read** token for your own account (Settings → Access Tokens on
huggingface.co) to pass as `hf_token` — separate from the `API_KEY`
secret above:

```powershell
py -c "from gradio_client import Client; c = Client('DhanushkaGodage/feedback-service-inference', hf_token='<your HF read token>'); print(c.predict('[{\"role\": \"user\", \"content\": \"Say hello in one word.\"}]', '<your API_KEY secret>', api_name='/generate'))"
```

You should get back a short generated string. A wrong `api_key` raises a
`gradio_client` error instead, and a missing/wrong `hf_token` fails before
that (the Space itself won't be reachable).

Then set `SPACE_ID`, `API_KEY`, and `HF_SPACE_TOKEN` (the same HF read
token) in `services/feedback-service/.env` to point the deployed service
at it.

## Notes

- **Speed**: ZeroGPU allocates a real GPU only while `generate()` is
  running, so a warm request should take a few seconds instead of the
  30-90s the old CPU-basic setup needed. `@spaces.GPU(duration=120)` caps
  how long a single call may hold the GPU — raise it in `app.py` if
  generation is being cut off, lower it if you're hitting ZeroGPU's daily
  quota too fast.
- **Cold starts**: same as before — a Space on the free tier sleeps after
  inactivity and wakes on the next request, with an extra delay while it
  restarts. Separately, each ZeroGPU call itself has to wait for GPU
  allocation, so even a warm Space has a little more latency than a plain
  CPU call would per-request.
- **Daily GPU quota**: ZeroGPU usage on free accounts is rate-limited. If
  requests start failing with quota/queue errors, that's expected — retry
  later or reduce `max_new_tokens`/`duration` in `app.py`.
- **Updating the adapter**: after retraining and re-running the Colab push
  cell, the Space won't pick up the new adapter until its container
  restarts (it loads the model once at startup). Use **Settings → Restart
  this Space**, or just wait for it to sleep and wake on its own.
- **Updating the server code**: repeat step 3 (copy files, commit, push)
  whenever you change `app.py`/`requirements.txt` here — the Space
  rebuilds automatically on push.
