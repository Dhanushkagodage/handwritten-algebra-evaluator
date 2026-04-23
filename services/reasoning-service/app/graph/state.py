from typing import TypedDict, List, Dict

class GraphState(TypedDict):
    question: str
    student_answer: List[Dict]
    step_validity: List[Dict]