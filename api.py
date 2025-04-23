# api/main.py
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json

# Import the agents
from agents.cas_agent import CASAgent
from agents.solver_agent import MathSolverAgent
from agents.verification_agent import VerificationAgent
from core.math_toolbox import MathToolbox
from core.virtual_tool_manager import VirtualToolManager
from workflows.math_workflow import math_workflow
from utils.exceptions import StopException

# Create shared instances
math_toolbox = MathToolbox()
virtual_tool_manager = VirtualToolManager()

app = FastAPI(
    title="Math Problem Solver API",
    description="API for solving mathematical problems using different solving agents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input models
class ProblemRequest(BaseModel):
    problem: str


class VerificationRequest(BaseModel):
    problem: str
    proposed_solution: str


# Response models
class AgentSolutionResponse(BaseModel):
    solution: str
    error: Optional[str] = None


class TribunalResponse(BaseModel):
    problem: str
    solution: str
    is_verified: bool
    verification_result: str
    attempts: int
    used_virtual_tool: bool
    virtual_tool_info: Optional[str] = None
    from_cache: bool
    agent_solutions: Dict[str, Optional[str]]


# Helper function to initialize agents
def get_solver_agent():
    return MathSolverAgent(math_toolbox, virtual_tool_manager)


def get_verification_agent():
    return VerificationAgent(math_toolbox)


def get_cas_agent():
    return CASAgent()


# Silent callback handler that doesn't require Streamlit
class SilentCallbackHandler:
    def __init__(self):
        self.container = self
        self.text = ""

    def write(self, text):
        self.text += str(text) + "\n"

    def markdown(self, text):
        self.write(text)

    def success(self, text):
        self.write(f"SUCCESS: {text}")

    def error(self, text):
        self.write(f"ERROR: {text}")

    def info(self, text):
        self.write(f"INFO: {text}")

    def warning(self, text):
        self.write(f"WARNING: {text}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        pass

    def on_llm_new_token(self, token, **kwargs):
        pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        pass

    def on_tool_end(self, output, **kwargs):
        pass

    def on_agent_action(self, action, **kwargs):
        pass


@app.post("/api/tribunal", response_model=TribunalResponse,
          summary="Solve a math problem using the tribunal approach",
          description="Solves a mathematical problem using multiple agents (solver, verification, and CAS) and performs majority voting to determine the final answer.")
async def solve_with_tribunal(request: ProblemRequest):
    try:
        solver_agent = get_solver_agent()
        verification_agent = get_verification_agent()
        cas_agent = get_cas_agent()
        callback_handler = SilentCallbackHandler()

        result = math_workflow(
            problem=request.problem,
            solver_agent=solver_agent,
            verification_agent=verification_agent,
            cas_agent=cas_agent,
            vtm=virtual_tool_manager,
            callback_handler=callback_handler
        )

        return result
    except StopException:
        # This is normally used for user input in Streamlit
        raise HTTPException(status_code=400, detail="User input required but not supported in API mode")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving problem: {str(e)}")


@app.post("/api/cas", response_model=AgentSolutionResponse,
          summary="Solve a math problem using the CAS agent",
          description="Solves a mathematical problem using the Computer Algebra System agent which utilizes SymPy.")
async def solve_with_cas(request: ProblemRequest):
    try:
        cas_agent = get_cas_agent()
        callback_handler = SilentCallbackHandler()

        solution = cas_agent.solve_problem(request.problem, callback_handler)
        return {"solution": solution}
    except Exception as e:
        return {"solution": "", "error": f"Error: {str(e)}"}


@app.post("/api/solver", response_model=AgentSolutionResponse,
          summary="Solve a math problem using the solver agent",
          description="Solves a mathematical problem using the main solver agent which utilizes tools and virtual tools.")
async def solve_with_solver(request: ProblemRequest):
    try:
        solver_agent = get_solver_agent()
        callback_handler = SilentCallbackHandler()

        solution = solver_agent.solve_problem(request.problem, callback_handler)
        return {"solution": solution}
    except StopException:
        # This is normally used for user input in Streamlit
        raise HTTPException(status_code=400, detail="User input required but not supported in API mode")
    except Exception as e:
        return {"solution": "", "error": f"Error: {str(e)}"}


@app.post("/api/verification", response_model=Dict[str, Any],
          summary="Verify a solution to a math problem",
          description="Verifies a proposed solution to a mathematical problem using the verification agent.")
async def verify_solution(request: VerificationRequest):
    try:
        verification_agent = get_verification_agent()
        callback_handler = SilentCallbackHandler()

        is_verified, verification_result = verification_agent.verify_result(
            request.problem,
            request.proposed_solution,
            callback_handler
        )

        return {
            "is_verified": is_verified,
            "verification_result": verification_result,
            "problem": request.problem,
            "proposed_solution": request.proposed_solution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying solution: {str(e)}")


@app.get("/", summary="API Root", description="Returns basic information about the API.")
async def root():
    return {
        "name": "Math Problem Solver API",
        "version": "1.0.0",
        "endpoints": [
            "/api/tribunal - Solve with tribunal approach",
            "/api/cas - Solve with CAS agent only",
            "/api/solver - Solve with solver agent only",
            "/api/verification - Verify a proposed solution"
        ]
    }


# Exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)