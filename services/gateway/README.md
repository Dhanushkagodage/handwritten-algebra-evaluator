# Gateway — Integration Layer

One call runs the whole pipeline:

```
answer image(s) + marking scheme image
        │
        ▼
  Module 01  ocr-service        :8000   /extract-pages + /extract-marking-scheme
        │
        ▼   ← schema adaptation
  Module 02  reasoning-service  :8002   /api/v1/evaluate
        │
        ▼   ← schema adaptation
  Module 03  feedback-service   :8003   /api/v1/feedback
        │
        ▼
  graded, explained result
```

The gateway **imports nothing** from the three services and **modifies none of
them**. Each keeps its own venv, its own credentials, and its own standalone
terminal workflows — running the gateway is purely additive.

It also holds **no secrets of its own**: no `OPENAI_API_KEY`, no `HF_TOKEN`.
Each module owns its own.

## Quick start

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Then `http://127.0.0.1:8080/docs`, and `GET /health/services` to confirm all
three modules are reachable.

To start everything at once (all three services + the gateway + the frontend):

```powershell
powershell -ExecutionPolicy Bypass -File ..\..\scripts\dev.ps1 -WithFrontend
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/jobs` | Start an evaluation, return `202 {job_id}` immediately |
| `GET` | `/api/v1/jobs/{id}` | Poll: `{status, stage, stages[], warnings[], result, error}` |
| `DELETE` | `/api/v1/jobs/{id}` | Cancel a running evaluation |
| `POST` | `/api/v1/evaluate` | The same pipeline, synchronously (see the latency warning) |
| `GET` | `/health` | Gateway liveness, instant |
| `GET` | `/health/services` | Fan-out over all three modules; `?strict=1` for 503 when degraded |

Form fields (identical for both entry points):

| Field | Type | Default |
|---|---|---|
| `answer_images` | 1–5 files | required |
| `marking_scheme_image` | file | required |
| `question_text` | string | `""` — improves OCR accuracy |
| `ocr_mode` | `openai_vision` \| `local` | `openai_vision` |
| `use_math_ocr` | bool | `false` |
| `question_id` | string | which detected question to grade, e.g. `Q2` |
| `multi_question_policy` | `first` \| `all` \| `error` | `first` |

### Which one should I use?

**The job API.** A full run takes 60–90 seconds warm and 2–4 minutes when the
feedback model's Hugging Face Space is cold, and only the job API can report
which stage is running. `stage` takes the values `queued | ocr | reasoning |
feedback | done` — the middle three match the frontend's progress tracker keys
exactly.

`POST /api/v1/evaluate` exists for `curl`/`Invoke-RestMethod`, smoke tests, and
report screenshots. It holds the connection open for the whole pipeline.

A failed job still answers **200** with `status: "failed"` and `error.stage`
naming the module that failed, so a poller has one happy path.

## What the gateway actually does

Hop 1→2 is nearly free: OCR's `{"reasoning_input": ...}` merged with its
`{"marking_scheme": ...}` *is* reasoning-service's `EvaluationRequest`.

**Hop 2→3 is the real work**, and lives in `app/services/adapters.py`:

| reasoning emits | feedback needs | bridge |
|---|---|---|
| `step_id` | `step_number` | rename |
| *(nothing)* | `expression` | joined from the OCR steps by `step_id` |
| `status`: `correct`/`incorrect`/`partially_correct`/`unclear` | `validity`: `correct`/`partial`/`incorrect` | `partially_correct`→`partial`; `unclear`→`partial` if marks were awarded, else `incorrect` |
| `validity` (a **bool**) | `validity` (a **string**) | same name, different type — never passed through |
| `step_validation[].error` | `error_description` | joined by `step_id` |
| `method_detection.detected_method` | `detected_method` | with fallbacks |
| `total_marks` (marks **earned**) | `assigned_marks` | clamped to the scheme total |
| *(not echoed back)* | `question_text`, `marking_scheme` | carried by the gateway |

It also guards several failure modes that would otherwise be silent:

- **`total_marks: 0`** — the OCR prompt emits 0 when no total is visible on the
  scheme image. Unguarded this reaches the UI as a `0/0` division and renders
  `NaN%`. The gateway backfills from the step marks and warns.
- **Steps OCR extracted but reasoning never analysed** are kept and marked
  unmatched rather than silently vanishing from the results page.
- **Steps reasoning reported that OCR never extracted** fall back to the
  matched scheme step's expected expression, then to a visible placeholder —
  never an empty string, which the feedback prompt would render as `Step 3: `.
- **Duplicate `step_id`s** are deduped keeping the highest-scoring row.
- **Reasoning's deterministic fallback** answers HTTP 200 with uniformly low
  confidence; the gateway detects it and warns that marks may be unreliable.

Everything it had to guess about is reported in `warnings[]`.

## Gotchas worth knowing

- **ocr-service listens on :8000**, not :8001 as the repo-root README and
  `.env.example` claim. Its health route is `GET /`, not `/health`.
- **Run with a single worker.** The job store is in-process, so with two
  workers a job created on worker A 404s on worker B. `--reload` also wipes the
  store on every file save.
- **Never retry the feedback call.** feedback-service already retries the HF
  Space internally with a `(0, 5, 15, 30)s` ladder. `FEEDBACK_TIMEOUT_S=300`
  exists to stay above that; set it lower and the gateway 504s while the
  downstream request is still succeeding.
- **`/health/services` reporting feedback as `up` is not a guarantee** — that
  service's `/health` is a static dict. It proves the process is alive, not
  that its Space is awake.
- **`ocr_mode=local` is slow.** The first run downloads EasyOCR/pix2tex models
  and blocks a threadpool worker for minutes; raise `OCR_TIMEOUT_S` if you use it.
- The gateway always calls `/extract-pages` (never `/extract`) and renames
  uploads to `{run_id}_page{n}.jpg`, because ocr-service stores raw uploads
  under the uploaded filename stem — two concurrent runs both sending
  `answer.jpg` would otherwise overwrite each other.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

71 tests, **no services, no network, no credentials required** — the adapters
are pure functions, and the pipeline and job store are exercised against an
`httpx.MockTransport`. This is where a contract change between modules should
break first.

## Layout

```
app/
├── main.py          app factory, CORS, lifespan (httpx client + job sweeper)
├── config.py        pydantic-settings
├── api/             health.py, evaluate.py (sync), jobs.py (async), uploads.py
├── clients/         base.py (timeouts + error normalization), ocr/reasoning/feedback
├── core/            errors.py, handlers.py
├── schemas/         common.py + one mirror module per upstream service + gateway.py
└── services/        adapters.py (pure), pipeline.py, jobs.py
```

Upstream **request** models are strict, so a mapping mistake fails here with a
precise message. Upstream **response** models are lenient, so an extra field
from an LLM-driven service degrades into a warning rather than a 502.
