# math_solver/workflows/math_workflow.py
import time
from typing import Dict, Any
from config.settings import MAX_VERIFICATION_RETRIES
from utils.exceptions import StopException
from utils.logging_utils import setup_logger
import streamlit as st

logger = setup_logger("math_workflow")


def math_workflow(problem: str, solver_agent, verification_agent, vtm, callback_handler=None) -> Dict[str, Any]:
    """
    Execute the math problem solving workflow with retries:
    Math_solver_agent -> Validation_agent, with up to 3 retries if validation fails

    Args:
        problem: The math problem to solve
        solver_agent: Agent for solving math problems
        verification_agent: Agent for verifying solutions
        vtm: virtual_tool_manager
        callback_handler: Optional callback handler for streaming output

    Returns:
        Dict containing problem, solution, verification status, etc.
    """
    max_retries = MAX_VERIFICATION_RETRIES
    attempts = 0
    solution = None
    is_verified = False
    verification_result = None
    used_virtual_tool = False
    virtual_tool_hash = None
    virtual_tool_info = None
    waiting_for_input = False

    # First, check if we have a virtual tool for this problem
    virtual_tool_manager = vtm
    logger.info(f"Checking for virtual tools for problem: {problem}")
    virtual_tool = virtual_tool_manager.find_matching_virtual_tool(problem)

    if virtual_tool and callback_handler:
        callback_handler.container.write(f"Found matching virtual tool: {virtual_tool['name']}")
        virtual_tool_info = virtual_tool['name']

    # Check if we're waiting for user input
    for key in st.session_state.keys():
        if key.startswith("user_input_") and key.endswith("_timeout_counter"):
            waiting_for_input = True
            if callback_handler:
                callback_handler.container.write("Waiting for user to provide input...")
            break

    # If we're waiting for input, don't try to solve yet
    if waiting_for_input:
        raise StopException("Waiting for user input in workflow")

    while attempts < max_retries and not is_verified:
        attempts += 1

        # Solve the problem
        if callback_handler:
            callback_handler.container.write(f"Attempt {attempts}/{max_retries}: Solving the problem...")

        # If we have a virtual tool and this is the first attempt, try using it
        if virtual_tool and attempts == 1:
            if callback_handler:
                callback_handler.container.write(f"Using virtual tool {virtual_tool['name']} to solve the problem...")
            try:
                # Get the problem hash for tracking failures
                virtual_tool_hash = virtual_tool_manager.hash_problem(problem)

                # Call the virtual tool function
                fn_ptr = virtual_tool["function"]
                virtual_tool_solution = fn_ptr(input_str=problem, math_toolbox=solver_agent.toolbox)
                solution = virtual_tool_solution
                used_virtual_tool = True

                if callback_handler:
                    callback_handler.container.write(f"Virtual tool produced solution: {solution}")

                # Immediately verify the virtual tool result
                is_verified, verification_result = verification_agent.verify_result(
                    problem, solution, callback_handler
                )

                # If verification fails, mark this as a virtual tool failure and try again with standard solver
                if not is_verified:
                    if callback_handler:
                        callback_handler.container.write(
                            f"❌ Virtual tool solution failed verification. Recording failure.")

                    # Record the virtual tool failure
                    if virtual_tool_hash:
                        tool_removed = virtual_tool_manager.record_tool_failure(virtual_tool_hash)
                        if tool_removed and callback_handler:
                            callback_handler.container.write(
                                f"⚠️ Virtual tool {virtual_tool['name']} has been removed due to repeated failures."
                            )

                    # Fall back to standard solver
                    if callback_handler:
                        callback_handler.container.write(f"Falling back to standard solver...")

                    # Clear the virtual tool solution so we use the standard solver next
                    solution = None
                    used_virtual_tool = False
                    virtual_tool = None
                    continue
            except StopException:
                # Propagate user input exceptions
                raise
            except Exception as e:
                if callback_handler:
                    callback_handler.container.write(
                        f"Virtual tool failed with error: {str(e)}. Falling back to regular solver..."
                    )
                logger.error(f"Virtual tool error: {str(e)}")

                # Record the virtual tool failure
                if virtual_tool_hash:
                    tool_removed = virtual_tool_manager.record_tool_failure(virtual_tool_hash)
                    if tool_removed and callback_handler:
                        callback_handler.container.write(
                            f"⚠️ Virtual tool {virtual_tool['name']} has been removed due to repeated failures."
                        )

                # Fall back to the regular solver if the virtual tool fails
                solution = None
                used_virtual_tool = False
                virtual_tool = None
                continue

        # If we don't have a solution yet (either no virtual tool or it failed), use the regular solver
        if solution is None:
            try:
                solution = solver_agent.solve_problem(problem, callback_handler)
                used_virtual_tool = False
            except StopException:
                # Propagate user input exceptions
                raise

        # Verify the solution if we haven't already
        if not is_verified:
            if callback_handler:
                callback_handler.container.write(f"Verifying solution from attempt {attempts}...")

            is_verified, verification_result = verification_agent.verify_result(
                problem, solution, callback_handler
            )

            if is_verified:
                if callback_handler:
                    callback_handler.container.write(f"✅ Solution verified on attempt {attempts}!")
                break
            elif attempts < max_retries:
                if callback_handler:
                    callback_handler.container.write(f"❌ Verification failed on attempt {attempts}. Trying again...")

                # If we used a virtual tool and it failed verification, record the failure
                if used_virtual_tool and virtual_tool_hash:
                    tool_removed = virtual_tool_manager.record_tool_failure(virtual_tool_hash)
                    if tool_removed and callback_handler:
                        callback_handler.container.write(
                            f"⚠️ Virtual tool {virtual_tool['name']} has been removed due to repeated failures."
                        )

                    # Don't try to use this virtual tool again
                    virtual_tool = None
                    used_virtual_tool = False

    # Ensure we have a result, even if verification failed
    if not is_verified and not verification_result:
        verification_result = "No verification result available."

    # Make sure we have a solution string, even if it's an error message
    if not solution:
        solution = "Failed to produce a solution."

    result = {
        "problem": problem,
        "solution": solution,
        "is_verified": is_verified,
        "verification_result": verification_result,
        "attempts": attempts,
        "used_virtual_tool": used_virtual_tool,
        "virtual_tool_info": virtual_tool_info
    }

    return result