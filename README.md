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
┌─────────────────────────────────────┐
│  Handwritten Answer Image + Question│
└────────────┬────────────────────────┘
             │
      ┌──────▼──────┐
      │ OCR Service │  (Module 01)
      │   :8001     │  Extract steps
      └──────┬──────┘
             │
      ┌──────▼────────────┐
      │ Reasoning Service │  (Module 02)
      │    :8002          │  Validate + Mark
      └──────┬────────────┘
             │
      ┌──────▼──────────────┐
      │ Feedback Service    │  (Module 03)
      │   :8003             │  Generate feedback
      └──────┬──────────────┘
             │
┌────────────▼──────────────────────────┐
│ Final Score + Step-by-Step Feedback   │
│ + Improvement Suggestions              │
└───────────────────────────────────────┘

React Frontend (Vite) :5173
        ↕
   [All Services]
```

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

**Entry Point:** `services/ocr-service/app/main.py`  
**Port:** `8001`

---

### **Module 02 — Multi-Agent Reasoning & Stepwise Marking** (`reasoning-service`)
**Responsibility:** Validate each step, detect solution method, and assign marks

| Component | Technology |
|---|---|
| Orchestration | LangGraph (multi-agent workflow) |
| LLM Backend | OpenAI GPT-4o (via LangChain) |
| Worker Agents | 3 parallel agents: Step Correctness, Method Detection, Scheme Matching |
| Supervisor Agent | Aggregates agent outputs and allocates final marks |
| Math Validation | SymPy (algebraic correctness checking) |

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

### 2️⃣ Setup Each Service

#### **OCR Service (Module 01)**
```bash
cd services/ocr-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env if needed
uvicorn app.main:app --reload --port 8001
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

### 3️⃣ Setup Frontend

```bash
cd frontend
npm install
npm run dev  # Runs on http://localhost:5173
```

### 4️⃣ Fine-tune Feedback Service (Optional)

LoRA training runs on free Google Colab/Kaggle (a T4 GPU), not locally — see
`services/feedback-service/app/training/COLAB.md` for the copy-paste cells
(dataset generation, training, and pushing the adapter to Hugging Face Hub).

---

## 📡 API Endpoints

### **OCR Service** (:8001)

**POST** `/api/v1/ocr`
- **Input:** Multipart form-data with image file
- **Output:** `{ question_text, answer_text, student_steps[], linked_question }`

```bash
curl -X POST http://localhost:8001/api/v1/ocr \
  -F "file=@answer.png"
```

**GET** `/health` — Service status

---

### **Reasoning Service** (:8002)

**POST** `/api/v1/analyze`
- **Input:** `{ question_text, student_steps[], marking_scheme[], total_marks }`
- **Output:** `{ step_analysis[], detected_method, assigned_marks, total_marks }`

```json
{
  "question_text": "Solve x^2 - 5x + 6 = 0",
  "student_steps": [
    {"step_number": 1, "expression": "x^2 - 5x + 6 = 0"},
    {"step_number": 2, "expression": "(x-2)(x-3) = 0"}
  ],
  "marking_scheme": [
    {"step_number": 1, "expected_expression": "x^2 - 5x + 6 = 0", "marks": 1},
    {"step_number": 2, "expected_expression": "(x-2)(x-3) = 0", "marks": 2}
  ],
  "total_marks": 5
}
```

**GET** `/health` — Service status

---

### **Feedback Service** (:8003)

**POST** `/api/v1/feedback`
- **Input:** Reasoning service output + marking scheme
- **Output:** `{ final_score, total_marks, step_feedback[], overall_feedback, improvement_suggestions[] }`

```json
{
  "question_text": "Solve x^2 - 5x + 6 = 0",
  "student_steps": [
    {"step_number": 1, "expression": "...", "is_correct": true, "marks_awarded": 1},
    {"step_number": 2, "expression": "...", "is_correct": true, "marks_awarded": 2}
  ],
  "detected_method": "factorization",
  "assigned_marks": 3,
  "total_marks": 5,
  "marking_scheme": [...]
}
```

**GET** `/health` — Service status

---

## 🗂️ Project Structure

```
handwritten-algebra-evaluator/
├── README.md                          # This file
├── .env.example                       # Root environment template
├── .gitignore                         # Git ignore rules
│
├── frontend/                          # React + Vite (npm run dev)
│   ├── src/
│   │   ├── App.tsx                    # Main app component
│   │   ├── main.tsx                   # Entry point
│   │   ├── components/                # React components (Navbar, etc.)
│   │   ├── pages/                     # Pages (Home, Evaluate, Results)
│   │   ├── lib/api.ts                 # Axios API clients
│   │   └── store/                     # Zustand state management
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── services/
│   ├── ocr-service/                   # Module 01 (Empty code, ready for team member)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── routers/ocr.py
│   │   │   ├── models/schemas.py
│   │   │   └── services/
│   │   ├── requirements.txt
│   │   └── .env.example
│   │
│   ├── reasoning-service/             # Module 02 (Empty code, ready for team member)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── routers/reasoning.py
│   │   │   ├── models/schemas.py
│   │   │   └── agents/
│   │   ├── requirements.txt
│   │   └── .env.example
│   │
│   └── feedback-service/              # Module 03 (Fully implemented)
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
│       │   └── serving/hf_space/      # Hugging Face Space (Docker) — serves base model + adapter
│       │       ├── app.py, Dockerfile, requirements.txt
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

### Root `.env.example`
```env
# --- OCR Service (Module 01) ---
OCR_SERVICE_URL=http://localhost:8001
OCR_PORT=8001

# --- Reasoning Service (Module 02) ---
REASONING_SERVICE_URL=http://localhost:8002
REASONING_PORT=8002
OPENAI_API_KEY=your_openai_key_here
LLM_MODEL=gpt-4o

# --- Feedback Service (Module 03) ---
FEEDBACK_SERVICE_URL=http://localhost:8003
FEEDBACK_PORT=8003
# URL of the persistent Hugging Face Space serving the fine-tuned model,
# and the shared secret it expects as X-API-Key — see
# services/feedback-service/app/serving/hf_space/DEPLOY.md.
SPACE_API_URL=https://dhanushkagodage-feedback-service-inference.hf.space/generate
API_KEY=your_shared_space_api_key_here

# Training-only (used by app/training/train.py inside Colab/Kaggle, not by
# the deployed service)
LORA_ADAPTER_DIR=./lora-adapter
WANDB_PROJECT=handwritten-algebra-evaluator
WANDB_RUN_NAME=qwen25-3b-feedback-lora
```

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
