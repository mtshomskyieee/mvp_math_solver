# math_solver/workflows/math_workflow.py
import time
import re  # Add this import for regex functions
from typing import Dict, Any
from config.settings import MAX_VERIFICATION_RETRIES
from utils.exceptions import StopException
from utils.logging_utils import setup_logger
import streamlit as st
import os
import pickle

logger = setup_logger("math_workflow")

# Global cache for the workflow
_workflow_cache = {}


def math_workflow(problem: str, solver_agent, verification_agent, vtm, callback_handler=None) -> Dict[str, Any]:
    """
    Execute the math problem solving workflow with retries and caching:
    Math_solver_agent -> Validation_agent, with up to 3 retries if validation fails
    """
    global _workflow_cache

    # Simple normalization for cache lookup - just lowercase and strip whitespace
    cache_key = problem.strip().lower()

    # Check if this exact problem has been solved before
    if cache_key in _workflow_cache:
        # Now do an additional check to ensure the math expressions are truly equivalent
        # Extract all numbers from both the cache key and the current problem
        import re
        cached_problem = _workflow_cache[cache_key]['problem']

        # Extract numbers from both problems
        current_numbers = re.findall(r'[-+]?\d*\.?\d+', problem)
        cached_numbers = re.findall(r'[-+]?\d*\.?\d+', cached_problem)
        if current_numbers == cached_numbers:
            cached_result = _workflow_cache[cache_key]
            logger.info(f"Using cached result for problem: {problem}")

            if callback_handler:
                callback_handler.container.write("✅ Found cached solution for this exact problem!")
                callback_handler.container.write(f"Cached solution: {cached_result['solution']}")

            # Return the cached result but indicate it came from cache
            cached_result_copy = cached_result.copy()
            cached_result_copy['from_cache'] = True
            return cached_result_copy

    # Rest of the function remains the same...
    max_retries = MAX_VERIFICATION_RETRIES
    attempts = 0
    solution = None
    is_verified = False
    verification_result = None
    used_virtual_tool = False
    virtual_tool_hash = None
    virtual_tool_info = None
    waiting_for_input = False
    newly_created_virtual_tool = None
    failed_virtual_tools = set()

    # First, check if we have a virtual tool for this problem
    virtual_tool_manager = vtm
    logger.info(f"Checking for virtual tools for problem: {problem}")
    virtual_tool = virtual_tool_manager.find_matching_virtual_tool(problem)

    # Get current number of virtual tools to detect new ones later
    initial_tool_count = len(virtual_tool_manager.virtual_tools)
    problem_hash = virtual_tool_manager.hash_problem(problem)
    initial_has_tool = problem_hash in virtual_tool_manager.virtual_tools

    if virtual_tool and callback_handler:
        callback_handler.container.write(f"Found matching virtual tool: {virtual_tool['name']}")
        virtual_tool_info = virtual_tool['name']
        virtual_tool_hash = problem_hash

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
        if virtual_tool and attempts == 1 and virtual_tool_hash not in failed_virtual_tools:
            if callback_handler:
                callback_handler.container.write(f"Using virtual tool {virtual_tool['name']} to solve the problem...")
            try:
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
                        failed_virtual_tools.add(virtual_tool_hash)

                        if tool_removed and callback_handler:
                            callback_handler.container.write(
                                f"⚠️ Virtual tool {virtual_tool['name']} has been removed due to reaching {virtual_tool_manager.max_failures} failures."
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
                    failed_virtual_tools.add(virtual_tool_hash)

                    if tool_removed and callback_handler:
                        callback_handler.container.write(
                            f"⚠️ Virtual tool {virtual_tool['name']} has been removed due to reaching {virtual_tool_manager.max_failures} failures."
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

                # Check if a new virtual tool was created
                current_tool_hash = virtual_tool_manager.hash_problem(problem)
                if len(virtual_tool_manager.virtual_tools) > initial_tool_count or (
                        not initial_has_tool and current_tool_hash in virtual_tool_manager.virtual_tools
                ):
                    newly_created_virtual_tool = current_tool_hash
                    if callback_handler:
                        tool_name = virtual_tool_manager.virtual_tools[current_tool_hash]['name']
                        callback_handler.container.write(f"Created new virtual tool: {tool_name}")

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
                    # Record the failure and add to our set of failed tools
                    tool_removed = virtual_tool_manager.record_tool_failure(virtual_tool_hash)
                    failed_virtual_tools.add(virtual_tool_hash)

                    if tool_removed and callback_handler:
                        callback_handler.container.write(
                            f"⚠️ Virtual tool {virtual_tool['name']} has been removed due to reaching {virtual_tool_manager.max_failures} failures."
                        )

                    # Don't try to use this virtual tool again
                    virtual_tool = None
                    used_virtual_tool = False

                # If this was a newly created virtual tool and verification failed, delete it immediately
                if newly_created_virtual_tool and newly_created_virtual_tool in virtual_tool_manager.virtual_tools:
                    tool_name = virtual_tool_manager.virtual_tools[newly_created_virtual_tool]['name']
                    if callback_handler:
                        callback_handler.container.write(
                            f"⚠️ Removing newly created virtual tool {tool_name} due to verification failure."
                        )

                    # Delete the virtual tool
                    del virtual_tool_manager.virtual_tools[newly_created_virtual_tool]

                    # Also remove it from the successful sequences to prevent recreation
                    if newly_created_virtual_tool in virtual_tool_manager.successful_sequences:
                        del virtual_tool_manager.successful_sequences[newly_created_virtual_tool]

                    # Reset the tracking variable
                    newly_created_virtual_tool = None

    # After all attempts, if still not verified and we created a virtual tool, remove it
    if not is_verified and newly_created_virtual_tool and newly_created_virtual_tool in virtual_tool_manager.virtual_tools:
        tool_name = virtual_tool_manager.virtual_tools[newly_created_virtual_tool]['name']
        if callback_handler:
            callback_handler.container.write(
                f"⚠️ Removing newly created virtual tool {tool_name} due to final verification failure."
            )

        # Delete the virtual tool
        del virtual_tool_manager.virtual_tools[newly_created_virtual_tool]

        # Also remove it from the successful sequences to prevent recreation
        if newly_created_virtual_tool in virtual_tool_manager.successful_sequences:
            del virtual_tool_manager.successful_sequences[newly_created_virtual_tool]

    # If we reached max retries and the virtual tool failed all attempts, delete it completely
    # This helps prevent tools that consistently fail verification from persisting
    for failed_tool_hash in failed_virtual_tools:
        if failed_tool_hash in virtual_tool_manager.tool_failure_counts:
            if virtual_tool_manager.tool_failure_counts[failed_tool_hash] >= virtual_tool_manager.max_failures:
                if failed_tool_hash in virtual_tool_manager.virtual_tools:
                    tool_name = virtual_tool_manager.virtual_tools[failed_tool_hash]['name']
                    if callback_handler:
                        callback_handler.container.write(
                            f"⚠️ Permanently removing unreliable virtual tool {tool_name} after reaching maximum failure count of {virtual_tool_manager.max_failures}."
                        )

                    # Remove the tool completely
                    del virtual_tool_manager.virtual_tools[failed_tool_hash]

                    # Also remove from successful sequences
                    if failed_tool_hash in virtual_tool_manager.successful_sequences:
                        del virtual_tool_manager.successful_sequences[failed_tool_hash]

                    # Clear the failure count
                    del virtual_tool_manager.tool_failure_counts[failed_tool_hash]

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
        "virtual_tool_info": virtual_tool_info,
        "from_cache": False
    }

    # Only cache verified solutions
    if is_verified:
        # Create a simple cache key - just the normalized problem text
        _workflow_cache[cache_key] = result.copy()
        logger.info(f"Cached result for problem: {problem}")

        # Save virtual tools to CSV after successful verification
        virtual_tool_manager.save_virtual_tools_to_csv()

    return result


def save_workflow_cache():
    """Save the workflow cache to disk"""
    global _workflow_cache
    try:
        with open('workflow_cache.pkl', 'wb') as f:
            pickle.dump(_workflow_cache, f)
        logger.info(f"Saved workflow cache with {len(_workflow_cache)} entries")

        # Also save virtual tools to CSV
        from ui.main_app import get_virtual_tool_manager
        vtm = get_virtual_tool_manager()
        if vtm:
            vtm.save_virtual_tools_to_csv()

    except Exception as e:
        logger.error(f"Failed to save workflow cache: {e}")


def load_workflow_cache():
    """Load the workflow cache from disk"""
    global _workflow_cache
    if os.path.exists('workflow_cache.pkl'):
        try:
            with open('workflow_cache.pkl', 'rb') as f:
                _workflow_cache = pickle.load(f)
            logger.info(f"Loaded workflow cache with {len(_workflow_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to load workflow cache: {e}")
            _workflow_cache = {}
    else:
        _workflow_cache = {}


# Initialize the cache when the module is imported
load_workflow_cache()