"""
LangGraph Workflow — Parallel Multi-Agent Evaluation Pipeline
─────────────────────────────────────────────────────────────

Execution model:
  START
    │
    ├──[parallel]── step_validation_node
    ├──[parallel]── method_detection_node
    └──[parallel]── scheme_matching_node
                         │
                    supervisor_node   (fan-in: runs after all three complete)
                         │
                        END

The three specialist agents run concurrently via LangGraph's parallel branching.
The supervisor runs only after all three have written their outputs to state.
"""
import logging

from langgraph.graph import END, START, StateGraph

from app.agents.method_detection_agent import method_detection_agent
from app.agents.scheme_matching_agent import scheme_matching_agent
from app.agents.step_validation_agent import step_validation_agent
from app.agents.supervisor_agent import supervisor_agent
from app.graph.state import EvaluationState

logger = logging.getLogger(__name__)

# ── Node name constants ────────────────────────────────────────────────────────
NODE_STEP_VALIDATION = "step_validation"
NODE_METHOD_DETECTION = "method_detection"
NODE_SCHEME_MATCHING = "scheme_matching"
NODE_SUPERVISOR = "supervisor"


def build_graph():
    """
    Compile and return the LangGraph evaluation workflow.

    Topology:
    - START branches in parallel to all three specialist agents (fan-out)
    - All three agents feed into the supervisor (fan-in)
    - Supervisor → END
    """
    builder = StateGraph(EvaluationState)

    # ── Register agent nodes ───────────────────────────────────────────────────
    builder.add_node(NODE_STEP_VALIDATION, step_validation_agent)
    builder.add_node(NODE_METHOD_DETECTION, method_detection_agent)
    builder.add_node(NODE_SCHEME_MATCHING, scheme_matching_agent)
    builder.add_node(NODE_SUPERVISOR, supervisor_agent)

    # ── Fan-out: START → all three parallel agents ─────────────────────────────
    builder.add_edge(START, NODE_STEP_VALIDATION)
    builder.add_edge(START, NODE_METHOD_DETECTION)
    builder.add_edge(START, NODE_SCHEME_MATCHING)

    # ── Fan-in: all three agents → supervisor ─────────────────────────────────
    builder.add_edge(NODE_STEP_VALIDATION, NODE_SUPERVISOR)
    builder.add_edge(NODE_METHOD_DETECTION, NODE_SUPERVISOR)
    builder.add_edge(NODE_SCHEME_MATCHING, NODE_SUPERVISOR)

    # ── Supervisor → END ──────────────────────────────────────────────────────
    builder.add_edge(NODE_SUPERVISOR, END)

    graph = builder.compile()
    logger.info("[Workflow] LangGraph evaluation pipeline compiled successfully")
    return graph
