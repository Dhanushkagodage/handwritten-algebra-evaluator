from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.agents.step_correctness import step_correctness_agent

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("step_checker", step_correctness_agent)

    builder.set_entry_point("step_checker")
    builder.add_edge("step_checker", END)

    return builder.compile()