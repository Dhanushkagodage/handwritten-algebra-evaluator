# AI-Based Automated Evaluation of Handwritten A/L Algebra Answers

**Team InnovateX** — IN4911 Comprehensive Group Project  
Supervisor: Dr. C.R.J. Amalraj

---

## 📋 Project Overview

This system automates the evaluation of handwritten Advanced Level (A/L) algebra exam answers using AI. Instead of checking only the final answer, it validates each algebraic step, detects the solution method, and generates clear, student-friendly feedback explaining marks and improvements.

### Problem It Solves

- ❌ Manual grading is time-consuming, inconsistent, and difficult to scale
- ❌ Most automated systems only check final answers, ignoring intermediate steps
- ❌ Existing tools fail to recognize alternative valid solution methods
- ❌ No explainable feedback — students don't know why marks were deducted

### Solution

✅ **Three-module pipeline** combining OCR, AI reasoning, and fine-tuned language models to produce fair, transparent, step-by-step evaluation with actionable student feedback.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────┐
│ React Frontend (Vite) :5173               │
│ answer image + marking scheme image       │
└────────────────────┬──────────────────────┘
                     │  ONE call
      ┌──────────────▼──────────────┐
      │ Gateway  :8080              │  Integration layer
      │ chains the three modules    │  + schema adaptation
      └──────────────┬──────────────┘
                     │
      ┌──────▼───────┴──────┐
      │ OCR Service   :8000 │  (Module 01)  Extract steps + marking scheme
      └──────┬──────────────┘
             │  schema adaptation
      ┌──────▼────────────┐
      │ Reasoning Service │  (Module 02)  :8002  Validate + Mark
      └──────┬────────────┘
             │  schema adaptation
      ┌──────▼──────────────┐
      │ Feedback Service    │  (Module 03)  :8003  Generate feedback
      └──────┬──────────────┘
             │
┌────────────▼──────────────────────────┐
│ Final Score + Step-by-Step Feedback   │
│ + Improvement Suggestions              │
└───────────────────────────────────────┘
```

The gateway is **additive**: it calls the three services over HTTP and imports
nothing from them, so each module keeps its own virtualenv, its own credentials,
and its own standalone terminal workflow. Everything each team member already
runs by hand still works exactly as before.

---

## 📦 Modules

### **Module 01 — OCR + Input Understanding** (`ocr-service`)
**Responsibility:** Extract handwritten algebra steps from exam paper images

| Component | Technology |
|---|---|
| Image Preprocessing | OpenCV (noise removal, skew correction, binarization) |
| Text Recognition | EasyOCR (English text extraction) |
| Math Expression Recognition | pix2tex / LaTeX-OCR (mathematical notation) |
| Output Format | Structured JSON with question, answer, and steps |

**Entry Point:** `services/ocr-service/app.py` (a flat app — run it as `uvicorn app:app`)  
**Port:** `8000`

---

### **Module 02 — Multi-Agent Reasoning & Stepwise Marking** (`reasoning-service`)
**Responsibility:** Validate each step, detect solution method, and assign marks

| Component | Technology |
|---|---|
| Orchestration | LangGraph (multi-agent workflow) |
| LLM Backend | OpenAI GPT-4o (via LangChain) |
| Worker Agents | 3 parallel agents: Step Correctness, Method Detection, Scheme Matching |
| Supervisor Agent | Aggregates agent outputs and allocates final marks |

**Architecture:**
```
Input (steps, question, marking scheme)
    ↓
[Agent 1: Step Correctness] ──┐
[Agent 2: Method Detection]   ├─→ Supervisor Agent
[Agent 3: Scheme Matching]    │   (Mark Allocation)
    ↓
Output (step validity, method, marks)
```

**Entry Point:** `services/reasoning-service/app/main.py`  
**Port:** `8002`

---

### **Module 03 — Stepwise Feedback Generation using SLM** (`feedback-service`)
**Responsibility:** Generate clear, student-friendly explanations of errors and improvements

| Component | Technology |
|---|---|
| Base Model | Qwen2.5-3B-Instruct (lightweight, efficient) |
| Fine-tuning Method | LoRA (Low-Rank Adaptation via PEFT), adapter only — base model stays untouched |
| Training Framework | Hugging Face Transformers + TRL (SFTTrainer), run on free Colab/Kaggle T4 |
| Inference | HTTP call to a persistent Hugging Face Space that loads the base model + adapter from the Hub |
| Experiment Tracking | Weights & Biases (wandb) |

**Training + Serving Pipeline:**
```
Raw Annotations (teacher feedback + student errors)
    ↓
dataset.py (format to prompt-completion pairs)
    ↓
train.py (LoRA fine-tuning, on Colab)
    ↓
LoRA adapter pushed to a private Hugging Face Hub repo
    ↓
Hugging Face Space (app/serving/hf_space/) loads base model + adapter
    ↓
feedback_generator.py (calls the Space over HTTP)
```

**Entry Point:** `services/feedback-service/app/main.py`  
**Port:** `8003`

---

## 💻 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18 + TypeScript, Vite, Tailwind CSS, shadcn/ui |
| **Backend** | FastAPI (Python 3.9+) × 3 services |
| **LLM/AI** | OpenAI GPT-4o, Qwen2.5-3B-Instruct, LangGraph, LoRA/PEFT |
| **Database** | PostgreSQL (optional), Redis (optional) |
| **Storage** | Local filesystem |
| **Version Control** | Git + GitHub |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (for backend services)
- **Node.js 18+** (for frontend)
- **OpenAI API key** (for reasoning service)
- **Hugging Face account** (optional, for model hosting)

### 1️⃣ Clone & Setup

```bash
git clone <repo-url>
cd handwritten-algebra-evaluator
```

### Fastest path — start everything at once (Windows)

Once each service has a `.venv` and a `.env`, this launches all four services
plus the frontend, waits for them to come up, and prints the gateway's view of
the stack:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 -WithFrontend
# first-time setup of any missing virtualenvs: add -Bootstrap
# stop everything:  .\scripts\stop-dev.ps1
```

The sections below cover setting each service up by hand, which is still the
normal way to work on a single module.

### 2️⃣ Setup Each Service

#### **OCR Service (Module 01)**
```bash
cd services/ocr-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
uvicorn app:app --reload   # flat app.py -> http://127.0.0.1:8000/docs
```

#### **Reasoning Service (Module 02)**
```bash
cd services/reasoning-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
uvicorn app.main:app --reload --port 8002
```

#### **Feedback Service (Module 03)**
```bash
cd services/feedback-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set SPACE_API_URL and API_KEY to point at the deployed Hugging Face
# Space — see services/feedback-service/app/serving/hf_space/DEPLOY.md
uvicorn app.main:app --reload --port 8003
```

#### **Gateway (Integration Layer)**
```bash
cd services/gateway
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --port 8080 --workers 1   # single worker: the job store is in-process
```

Then `GET http://127.0.0.1:8080/health/services` to confirm the gateway can
reach all three modules. Full documentation: `services/gateway/README.md`.

### 3️⃣ Setup Frontend

```bash
cd frontend
npm install
npm run dev  # Runs on http://localhost:5173
```

The frontend calls only the gateway. By default it uses the Vite dev proxy
(`/api` → `:8080`), so no configuration is needed; set `VITE_GATEWAY_URL` in
`frontend/.env` only if the gateway runs somewhere else.

### 4️⃣ Fine-tune Feedback Service (Optional)

LoRA training runs on free Google Colab/Kaggle (a T4 GPU), not locally — see
`services/feedback-service/app/training/COLAB.md` for the copy-paste cells
(dataset generation, training, and pushing the adapter to Hugging Face Hub).

---

## 📡 API Endpoints

### **Gateway** (:8080) — the integrated pipeline

**POST** `/api/v1/jobs` — start an evaluation, return `202 {job_id}` immediately
**GET** `/api/v1/jobs/{id}` — poll `{status, stage, stages[], warnings[], result, error}`
**POST** `/api/v1/evaluate` — the same pipeline synchronously (holds the connection for 1–4 minutes)
**GET** `/health/services` — check all three modules at once

```bash
curl -X POST http://127.0.0.1:8080/api/v1/jobs \
  -F "answer_images=@answer.jpg" \
  -F "marking_scheme_image=@scheme.jpg" \
  -F "question_text=Solve x^2 - 5x + 6 = 0"
```

`stage` advances through `ocr → reasoning → feedback → done`, so a client can
show which module is working. See `services/gateway/README.md` for the full
reference and the schema-adaptation table.

---

### **OCR Service** (:8000)

**POST** `/extract` — one answer image → `{ reasoning_input: { question_text, student_steps[{step_id, content}], final_answer } }`
**POST** `/extract-pages` — up to five ordered images (`image_1`…`image_5`); may return `reasoning_inputs[]` when several questions are detected
**POST** `/extract-marking-scheme` — a scheme image → `{ marking_scheme: { total_marks, steps[] } }`
**GET** `/` — service status (this service has no `/health` route)

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -F "image=@answer.png" \
  -F "question_text=Solve x^2 - 5x + 6 = 0"
```

---

### **Reasoning Service** (:8002)

**POST** `/api/v1/evaluate` — full output, including per-step validation errors and the detected method
**POST** `/api/v1/evaluate/summary` — a compact human-readable summary
**GET** `/health` — service status

Input — note `reasoning_input` is **nested**, and steps use `step_id`/`content`:

```json
{
  "reasoning_input": {
    "question_text": "Solve x^2 - 5x + 6 = 0",
    "student_steps": [
      {"step_id": 1, "content": "x^2 - 5x + 6 = 0"},
      {"step_id": 2, "content": "(x-2)(x-3) = 0"}
    ],
    "final_answer": "x = 2, x = 3"
  },
  "marking_scheme": {
    "total_marks": 3,
    "steps": [
      {
        "step_no": 1,
        "description": "State the equation in standard form",
        "expected_expression": "x^2 - 5x + 6 = 0",
        "marks": 1
      },
      {
        "step_no": 2,
        "description": "Factorise the quadratic into two linear factors",
        "expected_expression": "(x-2)(x-3) = 0",
        "marks": 2
      }
    ]
  }
}
```

Output: `{ steps_analysis[{step_id, status, marks_awarded, ...}], total_marks, max_marks, percentage, summary, method_feedback, step_validation, method_detection, scheme_matching }`.

---

### **Feedback Service** (:8003)

**POST** `/api/v1/feedback`
- **Input:** reasoning output adapted to this service's shape + the marking scheme
- **Output:** `{ final_score, total_marks, step_feedback[], overall_feedback, improvement_suggestions[] }`

```json
{
  "question_text": "Solve x^2 - 5x + 6 = 0",
  "student_steps": [
    {"step_number": 1, "expression": "x^2 - 5x + 6 = 0", "validity": "correct", "marks_awarded": 1},
    {"step_number": 2, "expression": "(x-2)(x-3) = 0", "validity": "partial", "marks_awarded": 1}
  ],
  "detected_method": "factorisation",
  "assigned_marks": 2,
  "marking_scheme": {"total_marks": 3, "steps": []}
}
```

**GET** `/health` — service status

> **The reasoning and feedback contracts do not line up directly.** Reasoning
> emits `step_id`, a four-value `status` (`correct` / `incorrect` /
> `partially_correct` / `unclear`), and no `expression`; feedback needs
> `step_number`, a three-value `validity` (`correct` / `partial` /
> `incorrect`), and an expression. The gateway owns that translation — see the
> mapping table in `services/gateway/README.md`.

> All three services exchange the **same** marking scheme object:
> `{ total_marks, steps: [{ step_no, description, expected_expression, marks }] }`.
> `total_marks` lives inside it and is never repeated as a sibling field.

**GET** `/health` — Service status

---

## 🗂️ Project Structure

```
handwritten-algebra-evaluator/
├── README.md                          # This file
├── .env.example                       # Root environment template
├── .gitignore                         # Git ignore rules
│
├── scripts/                           # dev.ps1 / stop-dev.ps1 / smoke.ps1
│
├── frontend/                          # React + Vite (npm run dev)
│   ├── src/
│   │   ├── App.tsx                    # Main app component
│   │   ├── main.tsx                   # Entry point
│   │   ├── types/api.ts               # TS mirrors of the backend contracts
│   │   ├── lib/api.ts                 # Gateway client (the only network file)
│   │   ├── hooks/useEvaluation.ts     # Start a job + poll it to completion
│   │   ├── components/                # Navbar, PipelineProgress, ErrorPanel, results/
│   │   ├── pages/                     # Home, Evaluate, Results
│   │   └── store/                     # Zustand state (sessionStorage-persisted)
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── services/
│   ├── gateway/                       # Integration layer (:8080)
│   │   ├── app/
│   │   │   ├── main.py                # FastAPI entry point
│   │   │   ├── api/                   # health, evaluate (sync), jobs (async)
│   │   │   ├── clients/               # httpx clients for the three services
│   │   │   ├── schemas/               # mirrors of each service's contract
│   │   │   └── services/
│   │   │       ├── adapters.py        # the schema bridge (pure functions)
│   │   │       ├── pipeline.py        # ocr -> reasoning -> feedback
│   │   │       └── jobs.py            # in-process job store
│   │   ├── tests/                     # 71 tests, no services needed
│   │   ├── requirements.txt
│   │   ├── .env.example
│   │   └── README.md
│   │
│   ├── ocr-service/                   # Module 01 (:8000, `uvicorn app:app`)
│   │   ├── app.py                     # flat FastAPI app — /extract, /extract-pages,
│   │   │                              #   /extract-marking-scheme
│   │   ├── src/                       # preprocess, segment, ocr_engine, openai_vision_ocr
│   │   ├── training/                  # pix2tex dataset prep + fine-tuning
│   │   ├── requirements.txt
│   │   └── .env.example
│   │
│   ├── reasoning-service/             # Module 02 (:8002)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── api/routes.py          # POST /api/v1/evaluate[/summary]
│   │   │   ├── schemas/               # input_schema.py, output_schema.py
│   │   │   ├── agents/                # step validation, method detection,
│   │   │   │                          #   scheme matching, supervisor
│   │   │   └── services/langgraph_flow.py
│   │   ├── tests/evaluation_suite/    # test cases + metrics/CSV recording
│   │   ├── requirements.txt
│   │   └── .env.example
│   │
│   └── feedback-service/              # Module 03 (:8003)
│       ├── app/
│       │   ├── main.py                # FastAPI entry point
│       │   ├── routers/feedback.py    # POST /api/v1/feedback
│       │   ├── models/schemas.py      # Pydantic schemas
│       │   ├── agents/
│       │   │   └── feedback_generator.py   # Calls the HF Space (base model + LoRA adapter)
│       │   ├── training/
│       │   │   ├── train.py           # LoRA fine-tuning script
│       │   │   ├── dataset.py         # Dataset preparation
│       │   │   └── data/              # Training data folder
│       │   └── serving/hf_space/      # Hugging Face Space (Gradio) — serves base model + adapter
│       │       ├── app.py, requirements.txt
│       │       └── DEPLOY.md          # One-time Space setup instructions
│       ├── requirements.txt
│       └── .env.example
│
├── shared/
│   └── models/
│       └── common.py                  # Shared Pydantic schemas
│
└── .git/                              # Git repository
```

---

## 🔐 Environment Variables

Each service reads its **own** `services/<name>/.env`. The root
`.env.example` is a reference copy of the whole set — see it for the full list.

| Service | Port | Needs |
|---|---|---|
| Gateway | 8080 | service URLs and timeouts only — **no secrets** |
| OCR (Module 01) | 8000 | `OPENAI_API_KEY`, `OPENAI_VISION_MODEL` |
| Reasoning (Module 02) | 8002 | `OPENAI_API_KEY` — **it fails at import without one** |
| Feedback (Module 03) | 8003 | `SPACE_ID`, `API_KEY`, `HF_TOKEN` |
| Frontend | 5173 | `VITE_GATEWAY_URL` (optional — blank uses the Vite proxy) |

---

## 🧑‍💻 Team & Responsibilities

| Member | ID | Module | Responsibility |
|---|---|---|---|
| Wijesinghe W.D.A.C. | 214235T | Module 01 | OCR + Input Understanding |
| Godage S.S.D. | 214069L | Module 02 | Multi-Agent Reasoning + Marking |
| Udayanga M.S.K. | 214213B | Module 03 | Stepwise Feedback Generation (SLM) |

---

## 📚 Key Features

✅ **Step-by-Step Validation** — Each algebraic step is independently validated  
✅ **Alternative Solution Methods** — Recognizes multiple valid approaches  
✅ **Marking Scheme Alignment** — Compares against teacher-defined rubrics  
✅ **Explainable Feedback** — Clear, student-friendly explanations  
✅ **Lightweight SLM** — Qwen2.5-3B-Instruct fine-tuned with LoRA  
✅ **Microservices Architecture** — Independent, scalable modules  

---

## 🔬 Research Gaps Addressed

### Module 01 — OCR
- Mixed-content handwritten OCR (text + math together)
- Exam-paper-specific preprocessing (noisy, skewed, photocopied documents)
- Handwritten symbol ambiguity (context-aware correction)

### Module 02 — Reasoning
- Lack of step-level reasoning analysis in automated systems
- No support for multiple solution methods
- Weak reasoning reliability in single-model outputs → Multi-agent validation

### Module 03 — Feedback
- Lack of explainable feedback in grading systems
- Poor quality educational feedback (too general or difficult)
- Lack of lightweight, deployable feedback systems

---

## 🧪 Testing

Each service exposes a `/health` endpoint:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

---

## 📖 References

- **Project Proposal:** See `Team InnovateX (1).pdf` in project root
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **PEFT (LoRA):** https://github.com/huggingface/peft
- **Qwen2.5 Models:** https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- **Hugging Face Spaces (Docker SDK):** https://huggingface.co/docs/hub/spaces-sdks-docker

---

## ⚙️ System Requirements

- **CPU:** 4+ cores
- **RAM:** 16GB+ (for running all 3 services + frontend simultaneously)
- **GPU:** Not required locally — feedback-service inference runs on a free Hugging Face Space; a GPU (or free Colab/Kaggle T4) is only needed transiently for LoRA training

---

## 📝 License

Academic research project — University of Colombo, 2026

---

## 🤝 Support

For questions or issues, contact supervisor: **Dr. C.R.J. Amalraj**

---

## 🎯 Next Steps for Team Members

1. **Module 01 (OCR):** Implement text + math OCR pipeline in `ocr-service/`
2. **Module 02 (Reasoning):** Wire up LangGraph agents in `reasoning-service/`
3. **Module 03 (Feedback):** Collect training data → Fine-tune Qwen2.5-3B-Instruct (LoRA) on Colab → push adapter to Hugging Face Hub → deploy the HF Space (see `services/feedback-service/app/serving/hf_space/DEPLOY.md`)

All service scaffolding is in place. Start coding!

---

**Generated:** April 13, 2026  
**Project Status:** Initiated ✅ | In Development 🔄
