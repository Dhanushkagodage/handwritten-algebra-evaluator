"""
Evaluation Controller module for Reasoning Service.
────────────────────────────────────────────────────
Encapsulates business logic, pipeline execution, fallback test case loading,
and response formatting for evaluation endpoints.
"""
import logging
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from app.core.exceptions import EmptySchemeError, EmptyStepsError, PipelineError
from app.schemas.input_schema import EvaluationRequest
from app.schemas.output_schema import EvaluationOutput, FriendlyEvaluation, FriendlyStep
from app.services.langgraph_flow import build_graph

logger = logging.getLogger(__name__)


class EvaluationController:
    """Controller handling algebra evaluation requests and pipeline execution."""

    def __init__(self):
        # Compile graph once during controller initialization
        self._graph = build_graph()

    def load_default_request(self, relative_path: Optional[str] = None) -> EvaluationRequest:
        """
        Load a fallback test case JSON when no request body is provided.
        Supports selecting a custom relative test case path if provided.
        """
        base_tests_dir = Path(__file__).resolve().parent.parent.parent / "tests"
        if relative_path:
            test_file = base_tests_dir / relative_path
        else:
            test_file = base_tests_dir / "al_algebra_cases" / "induction" / "q01_induction_perfect.json"

        logger.info("[EvaluationController] Loading test case from: %s", test_file)
        if not test_file.exists():
            raise FileNotFoundError(f"Test case file not found: {test_file}")

        with open(test_file, "r", encoding="utf-8") as f:
            return EvaluationRequest.model_validate(json.load(f))

    async def run_pipeline(self, request: EvaluationRequest) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Validate evaluation request, execute the LangGraph pipeline, and return
        (output_dict, state_dict, result_dict).
        """
        if not request.reasoning_input.student_steps:
            raise EmptyStepsError()
        if not request.marking_scheme.steps:
            raise EmptySchemeError()

        logger.info(
            "[EvaluationController] Executing pipeline — steps=%d, scheme_steps=%d",
            len(request.reasoning_input.student_steps),
            len(request.marking_scheme.steps),
        )

        state = {
            "question_text": request.reasoning_input.question_text,
            "student_steps": [s.model_dump() for s in request.reasoning_input.student_steps],
            "final_answer": request.reasoning_input.final_answer,
            "marking_scheme": request.marking_scheme.model_dump(),
        }

        try:
            result = await self._graph.ainvoke(state)
        except Exception as exc:
            logger.exception("[EvaluationController] Graph invocation failed: %s", exc)
            raise PipelineError(cause=str(exc))

        output = result.get("evaluation_output")
        if output is None:
            logger.error("[EvaluationController] evaluation_output missing from graph result")
            raise PipelineError(cause="No evaluation_output key in graph result")

        # Inject intermediate agent outputs for callers requiring deep inspection
        output["step_validation"] = result.get("step_validation_output")
        output["method_detection"] = result.get("method_detection_output")
        output["scheme_matching"] = result.get("scheme_matching_output")

        logger.info(
            "[EvaluationController] Evaluation complete — %.1f/%.1f marks (%.1f%%)",
            output["total_marks"],
            output["max_marks"],
            output["percentage"],
        )

        return output, state, result

    def build_friendly_response(self, output: Dict[str, Any], state: Dict[str, Any], result: Dict[str, Any]) -> FriendlyEvaluation:
        """Convert pipeline raw output dict into compact FriendlyEvaluation model."""
        step_content_map: Dict[int, str] = {
            s["step_id"]: s.get("content", "") for s in state["student_steps"]
        }
        method_detected: str = (
            result.get("method_detection_output", {}).get("detected_method", "undetermined")
        )
        friendly_steps = [
            FriendlyStep(
                step_number=step["step_id"],
                expression=step_content_map.get(step["step_id"], ""),
                validity=step["status"],
                marks_awarded=step["marks_awarded"],
            )
            for step in output["steps_analysis"]
        ]
        return FriendlyEvaluation(
            question_text=state["question_text"],
            detected_method=method_detected,
            assigned_marks=output["total_marks"],
            total_marks=output["max_marks"],
            student_steps=friendly_steps,
        )

    async def evaluate(self, request: Optional[EvaluationRequest] = None) -> EvaluationOutput:
        """Controller action for full detailed evaluation."""
        if request is None:
            request = self.load_default_request()

        output, state, result = await self.run_pipeline(request)
        return EvaluationOutput(**output)

    async def evaluate_summary(self, request: Optional[EvaluationRequest] = None) -> FriendlyEvaluation:
        """Controller action for compact friendly summary evaluation."""
        if request is None:
            request = self.load_default_request()

        output, state, result = await self.run_pipeline(request)
        return self.build_friendly_response(output, state, result)


# Singleton controller instance
evaluation_controller = EvaluationController()
