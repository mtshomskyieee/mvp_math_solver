# math_solver/ui/problem_solver.py
import streamlit as st
from config.settings import SAMPLE_PROBLEMS
from core.callbacks import StreamlitCallbackHandler
from workflows.math_workflow import math_workflow
from utils.exceptions import StopException

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


def solve_problem_section(problem, solver_agent, verification_agent, cas_agent):  # Add cas_agent
    """Render the problem solving section."""
    # Check if we're in the middle of waiting for user input
    waiting_for_input = any(key.startswith("user_input_") and key.endswith("_timeout_counter")
                            for key in st.session_state.keys())

    # Check if we need to clear the solving state
    if "clear_solving_state" in st.session_state and st.session_state.clear_solving_state:
        if "solving_in_progress" in st.session_state:
            del st.session_state.solving_in_progress
        st.session_state.clear_solving_state = False

        # Also clear any timeout counters to prevent issues
        for key in list(st.session_state.keys()):
            if key.endswith("_timeout_counter"):
                del st.session_state[key]

        st.rerun()

    # Only show the solve button if we're not already solving and not waiting for input
    if "solving_in_progress" not in st.session_state and not waiting_for_input:
        if st.button("Solve Problem"):
            if problem:
                # Mark that we're solving a problem
                st.session_state.solving_in_progress = True

                solution_container = st.container()
                # Create a callback handler for Streamlit
                callback_handler = StreamlitCallbackHandler(solution_container)

                with solution_container:
                    st.markdown("### Solution Process")
                    # Check for virtual tool first
                    virtual_tool_manager = st.session_state.virtual_tool_manager
                    virtual_tool = virtual_tool_manager.find_matching_virtual_tool(problem)
                    if virtual_tool:
                        st.write(f"Found a virtual tool that can solve this: {virtual_tool['name']}")

                    st.write("Starting solution workflow with validation and retries...")

                    try:
                        # Run the math workflow with the CAS agent
                        result = math_workflow(
                            problem=problem,
                            solver_agent=solver_agent,
                            verification_agent=verification_agent,
                            cas_agent=cas_agent,  # Add CAS agent
                            vtm=st.session_state.virtual_tool_manager,
                            callback_handler=callback_handler
                        )

                        # Store results in session state
                        st.session_state.workflow_result = result
                        # We're done solving
                        st.session_state.clear_solving_state = True
                        # Update the sidebar
                        st.session_state.sidebar_update_trigger = True
                        # Rerun to update the display
                        st.rerun()
                    except StopException:
                        # This exception will be thrown when we need user input
                        # Just let it pass - Streamlit will rerun and we'll continue
                        pass
                    except Exception as e:
                        st.error(f"Error solving problem: {str(e)}")
                        st.session_state.clear_solving_state = True
                        st.rerun()
            else:
                st.error("Please enter a problem to solve.")
    elif waiting_for_input:
        # If we're waiting for user input, display a message
        st.info("Please provide the requested input above to continue solving the problem.")
    else:
        # If we're in the middle of solving, continue the workflow
        solution_container = st.container()
        callback_handler = StreamlitCallbackHandler(solution_container)

        with solution_container:
            st.markdown("### Solution Process (Continuing)")

            try:
                # Continue the math workflow with the CAS agent
                result = math_workflow(
                    problem=problem,
                    solver_agent=solver_agent,
                    verification_agent=verification_agent,
                    cas_agent=cas_agent,  # Add CAS agent
                    vtm=st.session_state.virtual_tool_manager,
                    callback_handler=callback_handler
                )

                # Store results in session state
                st.session_state.workflow_result = result
                # We're done solving
                st.session_state.clear_solving_state = True
                # Update the sidebar
                st.session_state.sidebar_update_trigger = True
                # Rerun to update the display
                st.rerun()
            except StopException:
                # This exception will be thrown when we need user input
                # Just let it pass - Streamlit will rerun and we'll continue
                pass
            except Exception as e:
                st.error(f"Error solving problem: {str(e)}")
                st.session_state.clear_solving_state = True
                st.rerun()


# math_solver/ui/problem_solver.py - modify the display_agent_comparison function

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

                agent_data.append({
                    "Agent": agent.capitalize(),
                    "Solution": solution,
                    "Verification": verification,
                })

        if agent_data:
            # Convert to DataFrame for better display
            import pandas as pd
            df = pd.DataFrame(agent_data)
            st.table(df)

def display_solution_results():
    """Display solution results if available."""
    if st.session_state.get("workflow_result"):
        result = st.session_state.workflow_result
        solution_container = st.container()

        with solution_container:
            st.markdown("### Solution")
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
                        st.markdown(f"**{agent.capitalize()}**: {solution}")

            if result.get("used_virtual_tool", False):
                st.info(f"This problem was solved using a virtual tool!")

            if result["attempts"] > 1:
                st.info(f"The math solver needed {result['attempts']} attempts to reach a verified solution.")

        # Display what each agent had
        display_agent_comparison(result)

        # Reset the session state
        if st.button("Solve Another Problem"):
            # Clear workflow result
            st.session_state.workflow_result = None
            # Make sure solving state is cleared
            if "solving_in_progress" in st.session_state:
                del st.session_state.solving_in_progress
            # Rerun to update the UI
            st.rerun()