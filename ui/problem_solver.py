# math_solver/ui/problem_solver.py
import streamlit as st
from config.settings import SAMPLE_PROBLEMS
from core.callbacks import StreamlitCallbackHandler
from workflows.math_workflow import math_workflow
from utils.exceptions import StopException
import re


def problem_input_section():
    """Render the problem input section."""
    st.header("Enter a Math Problem")

    # init session state for input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Sample problems
    st.markdown("### Sample Problems")

    # Function to handle text area input change
    def on_text_area_change():
        if st.session_state.problem_input.strip() != "":
            st.session_state.selected_sample = "Enter your own"
        st.session_state.user_input = st.session_state.problem_input

    # Select box for sample problems
    selected_sample = st.selectbox(
        "Select a sample problem or enter your own below:",
        ["Enter your own"] + SAMPLE_PROBLEMS,
        key="selected_sample"
    )

    # Logic for text area content based on selection
    if selected_sample == "Enter your own":
        problem = st.text_area(
            "Enter your math problem:",
            value=st.session_state.user_input,
            height=100,
            key="problem_input",
            on_change=on_text_area_change
        )
    else:
        st.session_state.user_input = ""  # Clear user input when sample selected
        problem = selected_sample
        st.text_area(
            "Selected problem:",
            selected_sample,
            height=100,
            key="problem_input",
            on_change=on_text_area_change
        )

    return problem


def solve_problem_section(problem, solver_agent, verification_agent, cas_agent):
    """Render the problem solving section."""
    # Check if we're in the middle of waiting for user input
    waiting_for_input = any(key.startswith("user_input_") and key.endswith("_timeout_counter")
                            for key in st.session_state.keys())

    # Add a "Clear Results" button if we have results
    if st.session_state.get("solution_history", []):
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Clear Results"):
                # Reset all state but don't rerun
                st.session_state.solution_history = []
                if "workflow_result" in st.session_state:
                    del st.session_state.workflow_result
                if "solving_in_progress" in st.session_state:
                    del st.session_state.solving_in_progress
                if "current_solution_id" in st.session_state:
                    del st.session_state.current_solution_id

    # Initialize solution history if it doesn't exist
    if "solution_history" not in st.session_state:
        st.session_state.solution_history = []

    # Create solution ID if we don't have one
    if "current_solution_id" not in st.session_state and not waiting_for_input:
        import uuid
        st.session_state.current_solution_id = str(uuid.uuid4())

    # Display all previous solution history
    for i, solution_entry in enumerate(st.session_state.solution_history):
        with st.container():
            st.markdown(f"### Solution {i + 1}: {solution_entry['problem']}")
            st.markdown(solution_entry["content"])
            st.markdown("---")

    # Check if we need to resume from a user input
    if waiting_for_input:
        st.info("Please provide the requested input above to continue solving the problem.")

    # Always show the solve button
    solve_button = st.button("Solve Problem")

    # Create a new container for the current solution
    solution_container = st.container()

    # Check if we should start a new solution
    if solve_button and problem:
        # Add a new entry to the solution history
        solution_id = st.session_state.current_solution_id
        new_solution = {
            "id": solution_id,
            "problem": problem,
            "content": f"### Working on: {problem}\n",
            "is_complete": False
        }
        st.session_state.solution_history.append(new_solution)

        # Set solving in progress flag
        st.session_state.solving_in_progress = True

    # Process the current solution if we're solving
    if "solving_in_progress" in st.session_state and st.session_state.solving_in_progress:
        # Get the current solution entry
        solution_idx = len(st.session_state.solution_history) - 1
        if solution_idx >= 0:
            current_solution = st.session_state.solution_history[solution_idx]

            # Display the current solution content
            with solution_container:
                st.markdown(f"### Working on: {current_solution['problem']}")

                # Create a callback handler that appends to our solution content
                class AppendingStreamlitCallbackHandler(StreamlitCallbackHandler):
                    def __init__(self, container, solution_entry, solution_idx):
                        super().__init__(container)
                        self.solution_entry = solution_entry
                        self.solution_idx = solution_idx
                        self.text = ""

                    def on_llm_start(self, serialized, prompts, **kwargs):
                        super().on_llm_start(serialized, prompts, **kwargs)
                        self._append_to_solution("Thinking...\n")

                    def on_llm_new_token(self, token, **kwargs):
                        self.text += token
                        self.container.write(self.text)
                        # Periodically update the solution entry
                        if len(token) > 20:
                            self._append_to_solution(self.text)
                            self.text = ""

                    def on_tool_start(self, serialized, input_str, **kwargs):
                        super().on_tool_start(serialized, input_str, **kwargs)
                        self._append_to_solution(f"Using tool: {serialized['name']} with input: {input_str}\n")

                    def on_tool_end(self, output, **kwargs):
                        super().on_tool_end(output, **kwargs)
                        self._append_to_solution(f"Tool output: {output}\n")

                    def on_agent_action(self, action, **kwargs):
                        super().on_agent_action(action, **kwargs)
                        self._append_to_solution(f"Agent action: {action.tool} with input: {action.tool_input}\n")

                    def _append_to_solution(self, text):
                        self.solution_entry["content"] += text
                        st.session_state.solution_history[self.solution_idx] = self.solution_entry

                # Create our custom callback handler
                callback_handler = AppendingStreamlitCallbackHandler(
                    container=st,
                    solution_entry=current_solution,
                    solution_idx=solution_idx
                )

                # Check for virtual tool first
                virtual_tool_manager = st.session_state.virtual_tool_manager
                virtual_tool = virtual_tool_manager.find_matching_virtual_tool(problem)

                if virtual_tool:
                    # Update the solution entry
                    current_solution["content"] += f"Found a virtual tool that can solve this: {virtual_tool['name']}\n"
                    st.session_state.solution_history[solution_idx] = current_solution

                    try:
                        # Get the virtual tool function
                        fn_ptr = virtual_tool["function"]

                        # Call the virtual tool function and capture its output
                        current_solution["content"] += f"Using virtual tool {virtual_tool['name']}...\n"
                        st.session_state.solution_history[solution_idx] = current_solution
                        st.write(f"Using virtual tool {virtual_tool['name']}...")

                        # Execute the virtual tool
                        result = fn_ptr(input_str=problem, math_toolbox=solver_agent.toolbox)

                        # Add the result to our output
                        current_solution["content"] += f"Virtual tool produced solution: {result}\n"
                        st.session_state.solution_history[solution_idx] = current_solution
                        st.write(f"Virtual tool produced solution: {result}")

                        # Verify the result
                        current_solution["content"] += "Verifying virtual tool solution...\n"
                        st.session_state.solution_history[solution_idx] = current_solution
                        st.write("Verifying virtual tool solution...")

                        is_verified, verification_result = verification_agent.verify_result(
                            problem, result, callback_handler=callback_handler
                        )

                        # Add verification result to our output
                        if is_verified:
                            verification_message = f"✅ Virtual tool solution verified: {verification_result}\n"
                            current_solution["content"] += verification_message
                            st.success(verification_message)
                        else:
                            verification_message = f"❌ Virtual tool solution failed verification: {verification_result}\n"
                            current_solution["content"] += verification_message
                            st.error(verification_message)

                        # Create workflow result
                        workflow_result = {
                            "problem": problem,
                            "solution": result,
                            "is_verified": is_verified,
                            "verification_result": verification_result,
                            "attempts": 1,
                            "used_virtual_tool": True,
                            "virtual_tool_info": virtual_tool['name'],
                            "from_cache": False
                        }

                        # Store results in session state
                        st.session_state.workflow_result = workflow_result

                        # Mark solution as complete
                        current_solution["is_complete"] = True
                        st.session_state.solution_history[solution_idx] = current_solution

                        # We're done solving
                        st.session_state.solving_in_progress = False

                        # Create a new solution ID for next time
                        import uuid
                        st.session_state.current_solution_id = str(uuid.uuid4())

                        # Add final solution to history
                        current_solution["content"] += f"**Final Answer: {result}**\n"
                        st.session_state.solution_history[solution_idx] = current_solution

                        # Update the sidebar silently
                        st.session_state.sidebar_update_trigger = True

                        # Direct return without rerun
                        return

                    except StopException:
                        # This exception will be thrown when we need user input
                        # Just let it pass and maintain our solution history
                        return
                    except Exception as e:
                        # If virtual tool fails, log the error and continue with regular solver
                        error_message = f"Virtual tool failed with error: {str(e)}. Falling back to regular solver...\n"
                        current_solution["content"] += error_message
                        st.session_state.solution_history[solution_idx] = current_solution
                        st.error(error_message)
                        # Continue to standard workflow

                # If no virtual tool or virtual tool failed, proceed with standard workflow
                current_solution[
                    "content"] += "Starting solution workflow with planning, execution, validation and retries...\n"
                st.session_state.solution_history[solution_idx] = current_solution
                st.write("Starting solution workflow with planning, execution, validation and retries...")

                try:
                    # Run the math workflow with all agents
                    result = math_workflow(
                        problem=problem,
                        solver_agent=solver_agent,
                        verification_agent=verification_agent,
                        cas_agent=cas_agent,
                        math_planner_agent=st.session_state.math_planner_agent,
                        plan_execution_agent=st.session_state.plan_execution_agent,
                        vtm=st.session_state.virtual_tool_manager,
                        callback_handler=callback_handler
                    )

                    # Add final solution to history
                    current_solution["content"] += f"**Final Answer: {result['solution']}**\n"
                    if result["is_verified"]:
                        current_solution[
                            "content"] += f"✅ Solution has been verified! ({result['attempts']} attempts)\n"
                    else:
                        current_solution[
                            "content"] += f"❌ Solution could not be verified after {result['attempts']} attempts.\n"
                    current_solution["content"] += f"{result['verification_result']}\n"

                    # Mark solution as complete
                    current_solution["is_complete"] = True
                    st.session_state.solution_history[solution_idx] = current_solution

                    # Store results in session state
                    st.session_state.workflow_result = result
                    # We're done solving
                    st.session_state.solving_in_progress = False

                    # Create a new solution ID for next time
                    import uuid
                    st.session_state.current_solution_id = str(uuid.uuid4())

                    # Update the sidebar
                    st.session_state.sidebar_update_trigger = True

                    # Direct return without rerun
                    return

                except StopException:
                    # This exception will be thrown when we need user input
                    # Just let it pass - maintain our solution history
                    return
                except Exception as e:
                    error_message = f"Error solving problem: {str(e)}\n"
                    current_solution["content"] += error_message
                    st.session_state.solution_history[solution_idx] = current_solution
                    st.error(error_message)
                    st.session_state.solving_in_progress = False
                    return
    elif problem and not st.session_state.get("solving_in_progress", False):
        # Display message when no solving is happening
        with solution_container:
            st.write("Click 'Solve Problem' to start solving.")

def display_solution_results():
    """Display solution results if available."""
    if st.session_state.get("workflow_result"):
        result = st.session_state.workflow_result

        st.markdown("### Solution")
        # Check if this was a tribunal decision
        if "tribunal" in result.get("verification_result", "").lower():
            # Display the tribunal solution directly
            tribunal_solution = result["solution"]
            st.markdown(f"🏛️ **Tribunal solution**: {tribunal_solution}")

            # Show the source agents for the tribunal decision
            tribunal_agents = re.search(r'tribunal vote \((.*?)\)', result.get("verification_result", ""))
            if tribunal_agents:
                st.markdown(f"*Agreement between: {tribunal_agents.group(1)}*")
        else:
            st.write(result["solution"])

        st.markdown("### Verification")
        if result["is_verified"]:
            st.success(f"✅ The solution has been verified as correct! (Attempts: {result['attempts']})")
        else:
            st.error(f"❌ The solution could not be verified after {result['attempts']} attempts.")

        st.write(result["verification_result"])

        # Display agent solutions if available
        if 'agent_solutions' in result:
            st.markdown("### Agent Solutions")
            for agent, solution in result['agent_solutions'].items():
                if solution:
                    # Use descriptive agent names
                    agent_name = agent.capitalize()
                    if agent == "plan_executor":
                        agent_name = "Plan Executor"
                    elif agent == "solver":
                        agent_name = "Math Solver"
                    elif agent == "validation":
                        agent_name = "Validation Agent"
                    elif agent == "cas":
                        agent_name = "CAS Agent"

                    st.markdown(f"**{agent_name}**: {solution}")

        # Show plan execution details if available
        if result.get("plan_execution_details"):
            with st.expander("Show Plan Execution Details", expanded=False):
                st.markdown("### Plan Execution Details")

                plan_details = result.get("plan_execution_details", {})

                # Show final result
                if "final_result" in plan_details:
                    st.markdown(f"**Final Result:** {plan_details['final_result']}")

                # Display intermediate results if available
                if "intermediate_results" in plan_details:
                    st.markdown("#### Step-by-Step Execution")
                    for step in plan_details.get("intermediate_results", []):
                        # Create a clean display for each step
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"**Step {step.get('step', '?')}:**")
                        with col2:
                            st.markdown(step.get('description', 'No description'))

                        # Show tool details in an indented area
                        if "tool" in step:
                            with st.container():
                                st.markdown(f"**Tool:** `{step.get('tool', 'Unknown')}`")
                                st.markdown(f"**Input:** `{step.get('input', 'None')}`")

                                if "result" in step:
                                    # Highlight successful results in green
                                    st.success(f"**Result:** {step.get('result', 'None')}")
                                elif "error" in step:
                                    # Highlight errors in red
                                    st.error(f"**Error:** {step.get('error', 'Unknown error')}")
                                elif "note" in step:
                                    # Show notes in blue info boxes
                                    st.info(step.get('note', ''))

                        # Add a separator between steps
                        st.markdown("---")

        if result.get("used_virtual_tool", False):
            st.info(f"This problem was solved using a virtual tool!")

        if result["attempts"] > 1:
            st.info(f"The math solver needed {result['attempts']} attempts to reach a verified solution.")

        # Display what each agent had
        display_agent_comparison(result)

def display_agent_comparison(result):
    """Display a comparison of all agent solutions."""
    if 'agent_solutions' in result and any(result['agent_solutions'].values()):
        st.markdown("### 🤖 Agent Solutions Comparison")

        # Create a comparison table
        agent_data = []
        for agent, solution in result['agent_solutions'].items():
            if solution:
                # Get verification information
                verification = "N/A"
                if agent == "solver" and result.get("verification_result"):
                    verification = result["verification_result"]
                elif agent == "plan_executor" and result.get("verification_result"):
                    verification = result["verification_result"]

                # Use more descriptive agent names in the table
                agent_display_name = agent.capitalize()
                if agent == "plan_executor":
                    agent_display_name = "Plan Executor"
                elif agent == "solver":
                    agent_display_name = "Math Solver"
                elif agent == "validation":
                    agent_display_name = "Validation Agent"
                elif agent == "cas":
                    agent_display_name = "CAS Agent"

                agent_data.append({
                    "Agent": agent_display_name,
                    "Solution": solution,
                    "Verification": verification,
                })

        if agent_data:
            # Convert to DataFrame for better display
            import pandas as pd
            df = pd.DataFrame(agent_data)
            st.table(df)

            # If we have plan data, offer to show it
            if result.get("plan_data"):
                if st.checkbox("Show Solution Plan Details"):
                    st.markdown("### 📝 Solution Plan")
                    plan = result["plan_data"]

                    st.markdown(f"**Problem Type:** {plan.get('problem_type', 'Not specified')}")

                    st.markdown("**Steps:**")
                    for i, step in enumerate(plan.get("steps", [])):
                        desc = step.get("description", "")
                        tool_info = step.get("tool_info", {})

                        if tool_info:
                            st.markdown(f"**Step {i + 1}:** {desc}")
                            st.markdown(f"- Tool: `{tool_info.get('tool', 'None')}`")
                            st.markdown(f"- Input: `{tool_info.get('input', 'None')}`")
                        else:
                            st.markdown(f"**Step {i + 1}:** {desc}")

                    # Show final step
                    final_step = plan.get("final_step", {})
                    if final_step:
                        st.markdown(f"**Final Step:** {final_step.get('description', '')}")
                        st.markdown(f"**Result Info:** {final_step.get('result_info', '')}")