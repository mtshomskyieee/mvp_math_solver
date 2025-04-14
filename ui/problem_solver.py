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
    if st.session_state.get("workflow_result"):
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Clear Results"):
                # Remove all results and reset state
                if "workflow_result" in st.session_state:
                    del st.session_state.workflow_result
                if "solving_in_progress" in st.session_state:
                    del st.session_state.solving_in_progress
                st.rerun()

    # Always show the solve button
    if st.button("Solve Problem"):
        if problem:
            # Mark that we're solving a problem
            st.session_state.solving_in_progress = True

            # Create a placeholder for solution process
            solution_process = st.empty()

            with solution_process.container():
                st.markdown("### Solution Process")
                # Check for virtual tool first
                virtual_tool_manager = st.session_state.virtual_tool_manager
                virtual_tool = virtual_tool_manager.find_matching_virtual_tool(problem)
                if virtual_tool:
                    st.write(f"Found a virtual tool that can solve this: {virtual_tool['name']}")

                st.write("Starting solution workflow with planning, execution, validation and retries...")

                # Create a callback handler for Streamlit
                callback_handler = StreamlitCallbackHandler(st)

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

                    # Store results in session state
                    st.session_state.workflow_result = result
                    # We're done solving
                    st.session_state.solving_in_progress = False
                    # Update the sidebar
                    st.session_state.sidebar_update_trigger = True
                    st.rerun()

                except StopException:
                    # This exception will be thrown when we need user input
                    # Just let it pass - Streamlit will rerun and we'll continue
                    pass
                except Exception as e:
                    st.error(f"Error solving problem: {str(e)}")
                    st.session_state.solving_in_progress = False
        else:
            st.error("Please enter a problem to solve.")

    elif waiting_for_input:
        # If we're waiting for user input, display a message
        st.info("Please provide the requested input above to continue solving the problem.")

    elif "solving_in_progress" in st.session_state and st.session_state.solving_in_progress:
        # If we're in the middle of solving, continue the workflow
        st.markdown("### Solution Process (Continuing)")
        st.write("Continuing solution process...")

        # Create a callback handler for Streamlit
        callback_handler = StreamlitCallbackHandler(st)

        try:
            # Continue the math workflow with all agents
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

            # Store results in session state
            st.session_state.workflow_result = result
            # We're done solving
            st.session_state.solving_in_progress = False
            # Update the sidebar
            st.session_state.sidebar_update_trigger = True
            st.rerun()

        except StopException:
            # This exception will be thrown when we need user input
            # Just let it pass - Streamlit will rerun and we'll continue
            pass
        except Exception as e:
            st.error(f"Error solving problem: {str(e)}")
            st.session_state.solving_in_progress = False


# In math_solver/ui/problem_solver.py
# Update the display_solution_results function to ensure plan execution output is shown

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