# math_solver/ui/main_app.py (modified)
import streamlit as st
import json
import time
from config.settings import SAMPLE_PROBLEMS
from core.callbacks import StreamlitCallbackHandler
from core.math_toolbox import MathToolbox
from core.virtual_tool_manager import VirtualToolManager
from agents.solver_agent import MathSolverAgent
from agents.verification_agent import VerificationAgent
from agents.cas_agent import CASAgent
from agents.math_planner_agent import MathPlannerAgent  # Import the new agents
from agents.plan_execution_agent import PlanExecutionAgent
from workflows.math_workflow import math_workflow
from ui.sidebar import render_sidebar
from ui.problem_solver import problem_input_section, solve_problem_section, display_solution_results

import atexit
from workflows.math_workflow import save_workflow_cache

from utils.logging_utils import setup_logger

logger = setup_logger("main_app")

# Register the save_workflow_cache function to run on exit
atexit.register(save_workflow_cache)


def initialize_session_state():
    """Initialize session state variables."""
    if "toolbox" not in st.session_state:
        st.session_state.toolbox = MathToolbox()

    if "virtual_tool_manager" not in st.session_state:
        st.session_state.virtual_tool_manager = VirtualToolManager()

        # Attempt to load existing vector store
        if hasattr(st.session_state.virtual_tool_manager, 'vector_store'):
            try:
                loaded = st.session_state.virtual_tool_manager.vector_store.load()
                if loaded:
                    logger.info("Loaded existing vector store")
                else:
                    # Migrate existing tools to vector store on first run
                    st.session_state.virtual_tool_manager.migrate_existing_tools_to_vector_store()
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")

    if "solver_agent" not in st.session_state:
        st.session_state.solver_agent = MathSolverAgent(
            st.session_state.toolbox,
            st.session_state.virtual_tool_manager
        )

    if "verification_agent" not in st.session_state:
        st.session_state.verification_agent = VerificationAgent(st.session_state.toolbox)

    if "cas_agent" not in st.session_state:
        st.session_state.cas_agent = CASAgent()

    # Initialize the new agents
    if "math_planner_agent" not in st.session_state:
        st.session_state.math_planner_agent = MathPlannerAgent()

    if "plan_execution_agent" not in st.session_state:
        st.session_state.plan_execution_agent = PlanExecutionAgent(st.session_state.toolbox)

    # Add session state variables for the process flow
    if "sidebar_update_trigger" not in st.session_state:
        st.session_state.sidebar_update_trigger = False

    if "current_problem" not in st.session_state:
        st.session_state.current_problem = None

    if "current_solution" not in st.session_state:
        st.session_state.current_solution = None

    # Add state variable to store evaluation results
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None


def get_virtual_tool_manager():
    """Get the virtual tool manager from session state."""
    if "virtual_tool_manager" in st.session_state:
        return st.session_state.virtual_tool_manager
    return None


def run_evaluation_section(solver_agent, verification_agent, cas_agent, math_planner_agent, plan_execution_agent):
    """Render the evaluation section."""
    st.header("Run Evaluation")

    if st.button("Run Evaluation on Sample Problems"):
        eval_container = st.container()

        with eval_container:
            st.write("Running evaluation on sample problems...")

            # Use the math_workflow for each problem in the evaluation
            results = []
            for idx, problem in enumerate(SAMPLE_PROBLEMS):
                st.write(f"Evaluating problem {idx + 1}/{len(SAMPLE_PROBLEMS)}: {problem}")

                # Time the solution
                start_time = time.time()
                result = math_workflow(
                    problem=problem,
                    solver_agent=solver_agent,
                    verification_agent=verification_agent,
                    cas_agent=cas_agent,
                    math_planner_agent=math_planner_agent,
                    plan_execution_agent=plan_execution_agent,
                    vtm=st.session_state.virtual_tool_manager
                )
                solution_time = time.time() - start_time

                # Add solution time to result
                result["solution_time"] = solution_time
                results.append(result)

            # Calculate statistics
            successful = sum(1 for r in results if r["is_verified"])
            success_rate = successful / len(SAMPLE_PROBLEMS) if SAMPLE_PROBLEMS else 0
            avg_solution_time = sum(r["solution_time"] for r in results) / len(results) if results else 0
            avg_attempts = sum(r["attempts"] for r in results) / len(results) if results else 0

            evaluation_results = {
                "results": results,
                "summary": {
                    "total_problems": len(SAMPLE_PROBLEMS),
                    "successful_solutions": successful,
                    "success_rate": success_rate,
                    "average_solution_time": avg_solution_time,
                    "average_attempts": avg_attempts
                }
            }

            # Store results in session state
            st.session_state.evaluation_results = evaluation_results

            # Force a rerun to update sidebar
            st.rerun()


def display_evaluation_results():
    """Display evaluation results if available."""
    if st.session_state.evaluation_results is not None:
        st.markdown("### Evaluation Results")
        summary = st.session_state.evaluation_results["summary"]
        st.json(json.dumps(summary, indent=2))

        st.markdown("### Detailed Results")
        for idx, result in enumerate(st.session_state.evaluation_results["results"]):
            st.markdown(f"**Problem {idx + 1}**: {result['problem']}")
            st.markdown(f"**Solution**: {result['solution']}")
            st.markdown(f"**Verified**: {'✅ Yes' if result['is_verified'] else '❌ No'}")
            st.markdown(f"**Attempts**: {result['attempts']}")
            st.markdown(f"**Time**: {result['solution_time']:.2f} seconds")

            # Display all agent solutions
            if 'agent_solutions' in result:
                st.markdown("**Agent Solutions**:")
                for agent, solution in result['agent_solutions'].items():
                    if solution:
                        st.markdown(f"- {agent.capitalize()}: {solution}")

            st.markdown("---")

        # Add a button to clear results if desired
        if st.button("Clear Evaluation Results"):
            st.session_state.evaluation_results = None
            st.rerun()


def app():
    """Main application function."""
    st.title("Math Problem Solver with Multi-Agent System")

    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_sidebar(st.session_state.toolbox, st.session_state.virtual_tool_manager)

    # Check if we need to resume from a user input
    if "solving_context" in st.session_state:
        context = st.session_state.solving_context
        solution_container = st.empty()

        with solution_container:
            st.markdown("### Solution Process (Resuming)")
            st.write("Continuing after receiving your input...")

            try:
                result = math_workflow(
                    problem=context["problem"],
                    solver_agent=context["solver_agent"],
                    verification_agent=context["verification_agent"],
                    cas_agent=st.session_state.cas_agent,
                    math_planner_agent=st.session_state.math_planner_agent,  # Add the new agents
                    plan_execution_agent=st.session_state.plan_execution_agent,
                    vtm=st.session_state.virtual_tool_manager,
                    callback_handler=context["callback_handler"]
                )

                # Store results in session state
                st.session_state.workflow_result = result

                # Clear the solving context
                del st.session_state.solving_context

                # Set flag to trigger sidebar updates on next rerun
                st.session_state.sidebar_update_trigger = True

                # Force a rerun to update sidebar
                st.rerun()
            except Exception as e:
                st.error(f"Error solving problem: {str(e)}")
                if "solving_context" in st.session_state:
                    del st.session_state.solving_context

    # Create two columns for the main layout
    left_col, right_col = st.columns([3, 1])

    # Left Column - Problem Input Area
    with left_col:
        problem = problem_input_section()
        solve_problem_section(
            problem,
            st.session_state.solver_agent,
            st.session_state.verification_agent,
            st.session_state.cas_agent
        )
        display_solution_results()

    # Right Column - Evaluation Area
    with right_col:
        run_evaluation_section(
            st.session_state.solver_agent,
            st.session_state.verification_agent,
            st.session_state.cas_agent,
            st.session_state.math_planner_agent,  # Add the new agents
            st.session_state.plan_execution_agent
        )
        display_evaluation_results()