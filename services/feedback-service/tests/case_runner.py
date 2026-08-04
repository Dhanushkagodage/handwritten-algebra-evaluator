"""
Shared machinery behind tests/test_case_01.py … test_case_10.py.

Each of those scripts is a one-liner that names its case number; everything
they actually do lives here, so the ten cases can't drift apart.

Two backends, selected with --backend or the BACKEND env var:

    space   FeedbackGenerator -> gradio_client -> Hugging Face Space (ZeroGPU).
            The default, and exactly what the deployed service and the
            original tests/test_feedback.py do. Needs SPACE_ID / API_KEY /
            HF_TOKEN. No GPU required on this machine.

    local   LocalFeedbackGenerator -> base model + LoRA adapter loaded on the
            Colab/Kaggle T4. Needs a CUDA runtime plus ADAPTER_PATH (and
            HF_TOKEN if the adapter repo is private). Costs no ZeroGPU quota.

Both backends share the same prompt builder, parser and validation, so the
printed FeedbackResponse has the same structure either way.

    python tests/test_case_02.py
    python tests/test_case_02.py --backend local
    BACKEND=local python tests/test_case_02.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent

# Make `app` importable when run as a plain script from the service root,
# the same way tests/test_feedback.py does.
sys.path.insert(0, str(_TESTS_DIR.parent))

from dotenv import load_dotenv

# Does not override variables already in the environment, so Colab's
# os.environ[...] assignments win over anything in a local .env file.
load_dotenv()

from app.models.schemas import FeedbackRequest, StepValidity

SAMPLE_DIR = _TESTS_DIR / "sample_inputs"
BACKENDS = ("space", "local")


def sample_path(case_number: int) -> Path:
    return SAMPLE_DIR / f"sample_input_{case_number:02d}.json"


def resolve_backend(argv=None) -> str:
    """--backend wins over the BACKEND env var, which wins over 'space'."""
    argv = list(sys.argv[1:] if argv is None else argv)
    backend = os.getenv("BACKEND", "space")

    if "--backend" in argv:
        i = argv.index("--backend")
        if i + 1 >= len(argv):
            raise SystemExit("--backend needs a value: " + " | ".join(BACKENDS))
        backend = argv[i + 1]

    backend = backend.strip().lower()
    if backend not in BACKENDS:
        raise SystemExit(f"unknown backend {backend!r} — expected one of: " + " | ".join(BACKENDS))
    return backend


def make_generator(backend: str):
    if backend == "local":
        # Imported here, not at module top level: it pulls in torch/peft,
        # which aren't installed for a space-backend run.
        from local_generator import LocalFeedbackGenerator

        return LocalFeedbackGenerator()

    from app.agents.feedback_generator import FeedbackGenerator

    return FeedbackGenerator()


def load_request(case_number: int) -> FeedbackRequest:
    with open(sample_path(case_number), "r", encoding="utf-8") as f:
        return FeedbackRequest.model_validate(json.load(f))


def check_invariants(request: FeedbackRequest, response) -> list:
    """The spec invariants tests/test_feedback.py asserts, as a list of failures.

    Returned rather than raised so run_all_cases.py can report on all ten
    instead of stopping at the first bad one.
    """
    failures = []

    if len(response.step_feedback) != len(request.student_steps):
        failures.append(
            f"expected one feedback block per student step "
            f"({len(request.student_steps)}), got {len(response.step_feedback)}"
        )

    for sf in response.step_feedback:
        if sf.validity != StepValidity.CORRECT and not sf.why_marks_reduced:
            failures.append(f"step {sf.step_number} lost marks but has no deduction explanation")

    if response.final_score != request.assigned_marks:
        failures.append(
            f"final_score {response.final_score} != assigned_marks {request.assigned_marks}"
        )
    if response.total_marks != request.marking_scheme.total_marks:
        failures.append(
            f"total_marks {response.total_marks} != "
            f"marking_scheme.total_marks {request.marking_scheme.total_marks}"
        )
    if not response.improvement_suggestions:
        failures.append("expected at least one improvement suggestion")

    return failures


async def run_case(case_number: int, generator, *, verbose: bool = True) -> list:
    """Generate feedback for one sample and check it. Returns the failures."""
    request = load_request(case_number)

    if verbose:
        print("=" * 70)
        print(f"CASE {case_number:02d} — {sample_path(case_number).name}")
        print("=" * 70)
        print(f"Question: {request.question_text}")
        print(f"Method  : {request.detected_method}")
        print(f"Score   : {request.assigned_marks} / {request.marking_scheme.total_marks}")
        print(f"Steps   : {len(request.student_steps)}\n")

    response = await generator.generate(request)

    if verbose:
        print("-" * 70)
        print("GENERATED FEEDBACK")
        print("-" * 70)
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))

    failures = check_invariants(request, response)

    if verbose:
        if failures:
            print(f"\nFAILED case {case_number:02d}:")
            for f in failures:
                print(f"  - {f}")
        else:
            print(f"\nCase {case_number:02d}: all assertions passed.")

    return failures


def main(case_number: int) -> None:
    """Entry point for the individual tests/test_case_NN.py scripts."""
    backend = resolve_backend()
    print(f"Backend : {backend}\n")

    async def _run():
        generator = make_generator(backend)
        await generator.load_model()
        return await run_case(case_number, generator)

    failures = asyncio.run(_run())
    sys.exit(1 if failures else 0)
