# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from math_professor_rag import Config, MathKnowledgeBase, MCPWebSearchTool, MathGuardrails, FeedbackSystem
from langchain_openai import ChatOpenAI
from jeebench_evaluator import JEEBenchEvaluator

import asyncio

# ---------------------------------------------------------
# Initialize FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="Math Professor AI API", version="1.0")

# Allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Initialize core components
# ---------------------------------------------------------
config = Config()
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=config.GROQ_API_KEY,
    model_name=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    max_tokens=config.LLM_MAX_TOKENS,
)

guardrails = MathGuardrails()
knowledge_base = MathKnowledgeBase()
knowledge_base.load_datasets()
mcp_search = MCPWebSearchTool(config.TAVILY_API_KEY)
feedback_system = FeedbackSystem()

# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    response: str
    rating: int
    comments: str = ""
    improved_response: str = None

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/status")
def get_status():
    return {
        "status": "ok",
        "model": config.LLM_MODEL,
        "knowledge_base_loaded": knowledge_base.data_loaded
    }

@app.post("/ask")
async def ask_math_question(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question not allowed.")

    # 1️⃣ Validate input (optional guardrails)
    try:
        if hasattr(guardrails, "validate_input"):
            valid, msg = guardrails.validate_input(question)
            if not valid:
                raise HTTPException(status_code=400, detail=msg)
    except Exception:
        pass  # Non-critical, continue

    # 2️⃣ Check for previously corrected human feedback
    exact_fb = feedback_system.get_feedback_for_question(question)
    if exact_fb and exact_fb.get("improved_response"):
        return {
            "question": question,
            "solution": exact_fb["improved_response"],
            "context_source": "feedback_improved",
            "note": "✅ Returned improved solution from feedback system",
        }

    # 3️⃣ Retrieve contextual knowledge
    kb_results = knowledge_base.search(question, k=3)
    if kb_results:
        context_source = "knowledge_base"
        context = "\n\n".join([r["content"] for r in kb_results])
    else:
        context_source = "web_search"
        web_results = mcp_search.search(question)
        context = "\n\n".join([str(r) for r in web_results.get("results", [])])

    # 4️⃣ Gather high-quality feedback examples (few-shot)
    few_shot_section = ""
    try:
        feedback_examples = feedback_system.get_feedback_examples(limit=5)
        if feedback_examples:
            few_shot_section = "### Past high-quality corrections (learn from these):\n\n"
            for i, fb in enumerate(feedback_examples, 1):
                sol = fb.get("improved_response") or fb.get("response", "")
                short = sol if len(sol) < 1000 else sol[:1000] + "..."
                few_shot_section += (
                    f"Example {i}:\nQuestion: {fb.get('question','')}\n"
                    f"Correct Solution:\n{short}\n---\n"
                )
    except Exception:
        pass

    # 5️⃣ Build the model prompt
    prompt = f"""
You are an expert mathematics professor. Use the examples and context below to answer carefully.

{few_shot_section}

Context source: {context_source}
{context}

Current Problem:
{question}

Instructions:
- Show detailed step-by-step reasoning.
- Be clear, structured, and concise.
- End your response with a line starting with "Final Answer:".
    """

    # 6️⃣ Query the model
    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # 7️⃣ Validate output (optional)
    try:
        if hasattr(guardrails, "validate_output"):
            ok, msg = guardrails.validate_output(answer)
            if not ok:
                return {
                    "question": question,
                    "solution": answer,
                    "context_source": context_source,
                    "warning": f"⚠️ Output validation warning: {msg}",
                }
    except Exception:
        pass

    return {
        "question": question,
        "solution": answer,
        "context_source": context_source,
    }

@app.post("/feedback")
def store_feedback(req: FeedbackRequest):
    success = feedback_system.store_feedback(
        question=req.question,
        response=req.response,
        rating=req.rating,
        comments=req.comments,
        improved_response=req.improved_response
    )
    if not success:
        raise HTTPException(status_code=500, detail="Feedback could not be saved")
    return {"message": "Feedback stored successfully"}

@app.get("/feedback/{question}")
def get_feedback_for_question(question: str):
    fb = feedback_system.get_feedback_for_question(question)
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return fb

@app.post("/benchmark")
async def run_benchmark(num_samples: int = 5):
    evaluator = JEEBenchEvaluator(llm, knowledge_base, mcp_search)
    metrics = evaluator.run_benchmark(num_samples=num_samples)
    return metrics
