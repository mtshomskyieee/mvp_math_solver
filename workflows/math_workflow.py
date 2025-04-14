# math_solver/workflows/math_workflow.py (modified)
import time
import re
import os
import pickle
from typing import Dict, Any
from config.settings import MAX_VERIFICATION_RETRIES
from utils.exceptions import StopException
from utils.logging_utils import setup_logger
import streamlit as st

logger = setup_logger("math_workflow")

# Global cache for the workflow
_workflow_cache = {}


def math_workflow(problem: str, solver_agent, verification_agent, cas_agent, math_planner_agent, plan_execution_agent,
                  vtm, callback_handler=None) -> Dict[str, Any]:
    """
    Execute the math problem solving workflow with multiple agents and majority voting:
    1. Math_planner_agent - Plans the solution steps
    2. Plan_execution_agent - Executes the plan
    3. Math_solver_agent - Alternative approach using tools directly
    4. Validation_agent - Validates the solution
    5. CAS_agent - Computer algebra system approach

    Use majority voting (agreement among agents) to determine the final answer.
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

    # Initialize variables
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

    # Solutions from each agent
    planner_solution = None
    executor_solution = None
    solver_solution = None
    validation_solution = None
    cas_solution = None
    plan_data = None

    while attempts < max_retries and not is_verified:
        attempts += 1

        # Solve the problem with all agents
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
                solver_solution = virtual_tool_solution
                used_virtual_tool = True

                if callback_handler:
                    callback_handler.container.write(f"Virtual tool produced solution: {solver_solution}")

                # Immediately verify the virtual tool result
                is_verified, verification_result = verification_agent.verify_result(
                    problem, solver_solution, callback_handler
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
                    solver_solution = None
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
                solver_solution = None
                used_virtual_tool = False
                virtual_tool = None
                continue

        # If we don't have a solution yet, solve with all agents
        if solver_solution is None:
            # 1. First use Math Planner Agent to create a plan
            try:
                if callback_handler:
                    callback_handler.container.write("Creating solution plan with Math Planner Agent...")

                # Get the available tools as a string for the planner
                tools_description = solver_agent.toolbox.get_tools_string()

                # Create the solution plan
                plan_result = math_planner_agent.plan_solution(problem, tools_description, callback_handler)
                plan_data = plan_result.get("structured_plan")

                if callback_handler:
                    callback_handler.container.write("Plan created successfully.")

            except StopException:
                # Propagate user input exceptions
                raise
            except Exception as e:
                logger.error(f"Math Planner Agent error: {str(e)}")
                if callback_handler:
                    callback_handler.container.write(f"Math Planner Agent error: {str(e)}")

            # 2. Use Plan Execution Agent to execute the plan
            if plan_data:
                try:
                    if callback_handler:
                        callback_handler.container.write("Executing plan with Plan Execution Agent...")

                    execution_result = plan_execution_agent.execute_plan(problem, plan_data, callback_handler)
                    executor_solution = execution_result.get("solution")
                    # Store the full execution result to preserve all details
                    plan_execution_details = execution_result

                    if callback_handler:
                        callback_handler.container.markdown(f"### Plan Execution Solution")
                        callback_handler.container.markdown(executor_solution)

                except StopException:
                    # Propagate user input exceptions
                    raise
                except Exception as e:
                    logger.error(f"Plan Execution Agent error: {str(e)}")
                    plan_execution_details = None
                    if callback_handler:
                        callback_handler.container.write(f"Plan Execution Agent error: {str(e)}")

            # 3. Use Math Solver Agent (original approach)
            try:
                if callback_handler:
                    callback_handler.container.write("Solving with Math Solver Agent...")

                solver_solution = solver_agent.solve_problem(problem, callback_handler)
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
            except Exception as e:
                logger.error(f"Math Solver Agent error: {str(e)}")
                if callback_handler:
                    callback_handler.container.write(f"Math Solver Agent error: {str(e)}")

            # 4. Use Validation Agent
            try:
                if callback_handler:
                    callback_handler.container.write("Solving with Validation Agent...")

                # Get the solution from validation agent
                # We'll need to extract it from its verification methods
                _, validation_text = verification_agent.verify_result(problem, "Unknown", callback_handler)
                validation_solution = _extract_verification_solution(validation_text)

                if callback_handler:
                    callback_handler.container.write(f"Validation Agent solution: {validation_solution}")

            except Exception as e:
                logger.error(f"Validation Agent error: {str(e)}")
                if callback_handler:
                    callback_handler.container.write(f"Validation Agent error: {str(e)}")

            # 5. Use CAS Agent
            try:
                if callback_handler:
                    callback_handler.container.write("Solving with CAS Agent...")

                cas_solution = cas_agent.solve_problem(problem, callback_handler)

                if callback_handler:
                    callback_handler.container.write(f"CAS Agent solution: {cas_solution}")
                    callback_handler.container.markdown(f"### 🧮 CAS Analysis Result\n{cas_solution}")

            except Exception as e:
                logger.error(f"CAS Agent error: {str(e)}")
                if callback_handler:
                    callback_handler.container.write(f"CAS Agent error: {str(e)}")

            # Now perform majority voting
            if callback_handler:
                callback_handler.container.write("Performing majority voting...")
            # In math_solver/workflows/math_workflow.py

            # Add this before the majority voting logic
            if callback_handler:
                callback_handler.container.write("Performing majority voting...")

            # Log normalized solutions for debugging
            logger.info("=== Normalized Solutions for Voting ===")
            normalized_solutions = {}
            if executor_solution:
                normalized_solutions["plan_executor"] = _normalize_solution(executor_solution)
                logger.info(f"Plan Executor: {executor_solution} -> {normalized_solutions['plan_executor']}")
            if solver_solution:
                normalized_solutions["solver"] = _normalize_solution(solver_solution)
                logger.info(f"Solver: {solver_solution} -> {normalized_solutions['solver']}")
            if validation_solution:
                normalized_solutions["validation"] = _normalize_solution(validation_solution)
                logger.info(f"Validation: {validation_solution} -> {normalized_solutions['validation']}")
            if cas_solution:
                normalized_solutions["cas"] = _normalize_solution(cas_solution)
                logger.info(f"CAS: {cas_solution} -> {normalized_solutions['cas']}")

            # Show normalized solutions in the UI
            if callback_handler:
                callback_handler.container.markdown("#### Normalized Solutions for Voting")
                for agent, norm_sol in normalized_solutions.items():
                    callback_handler.container.write(f"{agent}: {norm_sol}")

            # Count solutions that agree with each other
            solution_counts = {}
            for agent, sol in normalized_solutions.items():
                if sol is None:
                    logger.info(f"Skipping None solution from {agent}")
                    continue  # Skip None values
                if sol not in solution_counts:
                    solution_counts[sol] = {"count": 1, "agents": [agent]}
                else:
                    solution_counts[sol]["count"] += 1
                    solution_counts[sol]["agents"].append(agent)

            # Find majority solution (if any) - any solution with 2+ votes
            majority_solution = None
            majority_agents = []
            max_count = 0

            for sol, data in solution_counts.items():
                logger.info(f"Solution: {sol}, Count: {data['count']}, Agents: {', '.join(data['agents'])}")
                if data["count"] >= 2 and data["count"] > max_count:  # Find the solution with the most agreement
                    majority_solution = sol
                    majority_agents = data["agents"]
                    max_count = data["count"]

            logger.info(f"Identified majority solution: {majority_solution} with agents: {majority_agents}")

            if majority_solution:
                if callback_handler:
                    callback_handler.container.success(f"🏛️ Tribunal solution: {majority_solution}")
                    callback_handler.container.write(f"Agents that agree: {', '.join(majority_agents)}")

                logger.info(f"Majority solution found: {majority_solution}")
                logger.info(f"Agents that agree: {', '.join(majority_agents)}")

                # Use the majority solution
                solution = majority_solution
                is_verified = True  # Always mark as verified when we have majority agreement
                verification_result = f"Solution verified by tribunal vote ({', '.join(majority_agents)})"
            else:
                # No majority - prioritize plan execution agent if available, otherwise default to solver solution
                if executor_solution:
                    if callback_handler:
                        callback_handler.container.write(
                            "No tribunal majority found, using Plan Execution Agent solution with verification...")
                    solution = executor_solution
                else:
                    if callback_handler:
                        callback_handler.container.write(
                            "No tribunal majority found, using Math Solver Agent solution with verification...")
                    solution = solver_solution

                # Verify the solution
                is_verified, verification_result = verification_agent.verify_result(
                    problem, solution, callback_handler
                )
        else:
            # We already have a solution from virtual tool
            solution = solver_solution

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
        "from_cache": False,
        "plan_data": plan_data,
        "plan_execution_details": plan_execution_details if 'plan_execution_details' in locals() else None,
        "agent_solutions": {
            "plan_executor": executor_solution,
            "solver": solver_solution,
            "validation": validation_solution,
            "cas": cas_solution
        }
    }

    # Only cache verified solutions
    if is_verified:
        # Create a simple cache key - just the normalized problem text
        _workflow_cache[cache_key] = result.copy()
        logger.info(f"Cached result for problem: {problem}")

        # Save virtual tools to CSV after successful verification
        virtual_tool_manager.save_virtual_tools_to_csv()

    return result


# In math_solver/workflows/math_workflow.py

def _normalize_solution(solution_text):
    """Normalize a solution string for comparison - improved version."""
    if not solution_text:
        return None

    logger.info(f"Normalizing solution: {solution_text}")

    # Convert to lowercase and remove extra whitespace
    normalized = solution_text.lower().strip()

    # Check for explicit answer patterns first - more specific patterns first
    answer_patterns = [
        # For statements like "The answer to the problem is 62"
        r'answer\s*(?:to|is|:|=)\s*(?:the\s*problem)?\s*(?:is|:|=)?\s*(-?\d+\.?\d*)',
        # For statements like "The sum of 25 and 37 is 62"
        r'(?:sum|product|difference|quotient|result).*?(?:is|:|=)\s*(-?\d+\.?\d*)',
        # For statements like "25 + 37 equals 62"
        r'(?:equals|equal\s*to)\s*(-?\d+\.?\d*)',
        # For any numbers after specific keywords
        r'(?:is|:|=|gives|yields|evaluates\s*to)\s*(-?\d+\.?\d*)'
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                value = float(match.group(1))
                logger.info(f"Matched pattern '{pattern}' to extract: {value}")
                # Return as integer if it's a whole number
                if value.is_integer():
                    return str(int(value))
                # Otherwise return with limited precision
                return f"{value:.6f}".rstrip('0').rstrip('.')
            except Exception as e:
                logger.error(f"Error converting matched value: {e}")
                continue

    # If the entire string is just a number, use that (this helps with Validation and CAS agents)
    if re.match(r'^-?\d+\.?\d*$', normalized):
        try:
            value = float(normalized)
            logger.info(f"Using numeric string directly: {value}")
            if value.is_integer():
                return str(int(value))
            return f"{value:.6f}".rstrip('0').rstrip('.')
        except:
            pass

    # Fall back to extracting any numeric value - last resort
    numeric_match = re.search(r'(-?\d+\.?\d+)', normalized)
    if numeric_match:
        try:
            # Convert to float and format consistently
            value = float(numeric_match.group(1))
            logger.info(f"Extracted numeric value (fallback): {value}")
            # Return as integer if it's a whole number
            if value.is_integer():
                return str(int(value))
            # Otherwise return with limited precision
            return f"{value:.6f}".rstrip('0').rstrip('.')
        except:
            pass

    # If the entire string is just a number, use that
    if re.match(r'^-?\d+\.?\d*$', normalized):
        try:
            value = float(normalized)
            if value.is_integer():
                return str(int(value))
            return f"{value:.6f}".rstrip('0').rstrip('.')
        except:
            pass

    # For complex numbers, normalize format
    if 'i' in normalized:
        complex_match = re.search(r'([-+]?\d*\.?\d*)\s*\+\s*([-+]?\d*\.?\d*)i', normalized)
        if complex_match:
            real = float(complex_match.group(1) or 0)
            imag = float(complex_match.group(2) or 1)
            if real == 0:
                if imag == 1:
                    return "i"
                elif imag == -1:
                    return "-i"
                else:
                    return f"{imag}i"
            else:
                if imag == 1:
                    return f"{real}+i"
                elif imag == -1:
                    return f"{real}-i"
                elif imag > 0:
                    return f"{real}+{imag}i"
                else:
                    return f"{real}{imag}i"

    # Return the cleaned text if we couldn't extract a number
    logger.info(f"Could not extract numeric value, returning cleaned text: {normalized}")
    return normalized

def _extract_verification_solution(verification_text):
    """Extract solution from verification agent output."""
    if not verification_text:
        return None

    # Try to find the "Expected X" or "correct answer is X" pattern
    expected_match = re.search(r'expected\s+(.+?)(?:\.|$)', verification_text.lower())
    if expected_match:
        return expected_match.group(1).strip()

    correct_match = re.search(r'correct\s+answer\s+is\s+(.+?)(?:\.|$)', verification_text.lower())
    if correct_match:
        return correct_match.group(1).strip()

    # Look for EXACT_ANSWER format
    exact_match = re.search(r'EXACT_ANSWER:\s+(.+?)(?:\.|$)', verification_text)
    if exact_match:
        return exact_match.group(1).strip()

    return None


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


def clear_workflow_cache():
    """Clear the workflow cache completely."""
    global _workflow_cache
    _workflow_cache = {}
    logger.info("Workflow cache cleared")

    # Also try to remove the cache file if it exists
    if os.path.exists('workflow_cache.pkl'):
        try:
            os.remove('workflow_cache.pkl')
            logger.info("Removed workflow_cache.pkl file")
        except Exception as e:
            logger.error(f"Failed to remove workflow_cache.pkl file: {e}")

    return len(_workflow_cache)


# Initialize the cache when the module is imported
load_workflow_cache()