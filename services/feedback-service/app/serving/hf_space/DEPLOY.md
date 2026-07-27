# Deploying the inference Space (one-time setup)

This folder (`app.py`, `requirements.txt`, `Dockerfile`, `README.md`) is
the code for a **Hugging Face Space** — a separate, always-on Docker
container hosted by Hugging Face, independent of Colab. A Space is its own
git repository at `huggingface.co/spaces/<org>/<name>`, so this folder gets
pushed there directly; it isn't deployed via this GitHub repo.

You only need to do this once (or again if you want to change how the
server itself behaves — model swaps only need the Colab push step in
`app/training/COLAB.md`, not a redeploy here).

## 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Owner**: `DhanushkaGodage` (personal account) — so it's
   `huggingface.co/spaces/DhanushkaGodage/feedback-service-inference`.
3. **SDK**: **Docker** → **Blank**.
4. **Hardware**: free **CPU basic** (2 vCPU, 16 GB RAM).
5. **Visibility**: Public is fine — the app enforces its own `X-API-Key`
   check, so the model isn't callable by anyone who doesn't have the key,
   even though the Space URL itself is reachable.
6. Click **Create Space**.

## 2. Set the Space's secrets

In the Space → **Settings → Variables and secrets**, add these as
**secrets** (not public variables):

| Name              | Value                                                              |
|-------------------|---------------------------------------------------------------------|
| `HF_TOKEN`        | A **read**-scoped token with access to your private adapter repo   |
| `ADAPTER_REPO_ID` | `DhanushkaGodage/qwen25-feedback-lora` (from the Colab push step)  |
| `BASE_MODEL`      | `Qwen/Qwen2.5-3B-Instruct`                                          |
| `API_KEY`         | Any random string you generate — shared secret for calling `/generate` |

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

The Space builds the Docker image and starts the container automatically
(watch the **Logs** tab — first boot downloads the ~6 GB base model, which
can take several minutes). Once you see
`Loaded Qwen/Qwen2.5-3B-Instruct + adapter DhanushkaGodage/... on CPU (bfloat16)`
in the logs, test it:

```powershell
curl -X POST https://dhanushkagodage-feedback-service-inference.hf.space/generate `
  -H "Content-Type: application/json" `
  -H "X-API-Key: <your API_KEY secret>" `
  -d '{"messages": [{"role": "user", "content": "Say hello in one word."}]}'
```

You should get back `{"text": "..."}`. Also check
`https://dhanushkagodage-feedback-service-inference.hf.space/health`.

Then set `SPACE_API_URL` (with the `/generate` suffix) and `API_KEY` in
`services/feedback-service/.env` to point the deployed service at it.

## Notes

- **Speed**: free-tier CPU inference for a 3B model is slow — expect
  roughly 30-90 seconds per request depending on response length.
  `app.py` caps `max_new_tokens` at 300 to keep this bounded; lower it
  further in `app.py` if responses are consistently too slow.
- **Cold starts**: a public Space on the free tier sleeps after a period
  of inactivity and wakes on the next request (with an extra delay while
  it restarts). This is expected — there's no cost either way.
- **Updating the adapter**: after retraining and re-running the Colab push
  cell, the Space won't pick up the new adapter until its container
  restarts (it loads the model once at startup). Use **Settings → Restart
  this Space**, or just wait for it to sleep and wake on its own.
- **Updating the server code**: repeat step 3 (copy files, commit, push)
  whenever you change `app.py`/`requirements.txt`/`Dockerfile` here — the
  Space rebuilds automatically on push.
