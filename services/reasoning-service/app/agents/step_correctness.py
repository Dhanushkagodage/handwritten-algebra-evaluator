import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a mathematics evaluator.

Your task:
- Check each step of a student's solution
- Determine if each step is mathematically correct

Rules:
- Do NOT assign marks
- Do NOT skip steps
- Analyze step-by-step logically
- If unclear, mark as "unclear"
- If partially correct, mark as "partially_correct"

Output strictly in JSON format:
{
  "step_validity": [
    {
      "step": number,
      "status": "correct | incorrect | partially_correct | unclear",
      "reason": "short explanation"
    }
  ]
}
"""),

    ("human", """
Question:
{question}

Student Answer:
{student_answer}
""")
])

def step_correctness_agent(state):
    chain = prompt | llm

    response = chain.invoke({
        "question": state["question"],
        "student_answer": json.dumps(state["student_answer"], indent=2)
    })

    result = json.loads(response.content)

    return {
        "step_validity": result["step_validity"]
    }