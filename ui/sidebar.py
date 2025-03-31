# math_solver/ui/sidebar.py
import streamlit as st
from core.math_toolbox import MathToolbox
from core.virtual_tool_manager import VirtualToolManager


def render_sidebar(toolbox: MathToolbox, virtual_tool_manager: VirtualToolManager):
    """Render the sidebar components."""
    # # Sidebar info
    # st.sidebar.header("Agents")
    # st.sidebar.markdown("""
    # 1. Math Solver Agent - Uses tools to solve problems
    # 2. Verification Agent - Verifies solutions
    # 3. Virtual Tool Manager - Creates new tools from successful sequences
    # """)

    # Add toggle for tool errors
    st.sidebar.header("Settings")

    # Initialize the toggle state if it doesn't exist
    if "tool_errors_enabled" not in st.session_state:
        st.session_state.tool_errors_enabled = True  # Default to having errors enabled

    # Create the toggle button
    toggle_tools = st.sidebar.toggle("Add Tool Errors",
                                     value=st.session_state.tool_errors_enabled,
                                     help="Toggle between reliable and unreliable tools")

    # Check if the toggle state has changed
    if toggle_tools != st.session_state.tool_errors_enabled:
        st.session_state.tool_errors_enabled = toggle_tools

        # Apply the change to the toolbox
        if toggle_tools:
            toolbox.unset_all_tools_reliable()
            st.sidebar.info("Tool errors enabled. Some tools may now produce incorrect results.")
        else:
            toolbox.set_all_tools_reliable()
            st.sidebar.success("All tools set to reliable mode. No errors will be introduced.")

    # Tool usage statistics in sidebar
    st.sidebar.header("Tool Statistics")
    tool_stats = toolbox.get_stats()

    # Create columns for tool stats
    col1, col2 = st.sidebar.columns(2)

    for i, (tool_name, stats) in enumerate(tool_stats.items()):
        calls = stats["calls"]
        errors = stats["errors"]
        error_rate = errors / calls if calls > 0 else 0

        # Determine color based on error rate
        # Green if error rate is 0, otherwise red
        delta_color = "normal" if error_rate == 0 else "inverse"

        # Format error rate as percentage (only if not zero)
        if error_rate > 0:
            error_rate_display = f"{error_rate:.1%} err"
            delta_color = "inverse"  # Red for error rates > 0
        else:
            # For zero error rate, don't show any delta
            error_rate_display = f"{error_rate:.1%}"
            delta_color = 'off'  # delta_color only accepts: 'normal', 'inverse', or 'off'

        # Alternate between columns
        if i % 2 == 0:
            col1.metric(
                label=tool_name,
                value=f"{calls} calls",
                delta=error_rate_display,
                delta_color=delta_color
            )
        else:
            col2.metric(
                label=tool_name,
                value=f"{calls} calls",
                delta=error_rate_display,
                delta_color=delta_color
            )

    # Virtual tools in sidebar
    st.sidebar.header("Virtual Tools")
    virtual_tools = virtual_tool_manager.virtual_tools

    if not virtual_tools:
        st.sidebar.write("No virtual tools created yet.")
    else:
        for tool_id, tool in virtual_tools.items():
            # Get failure count if available
            failure_count = virtual_tool_manager.tool_failure_counts.get(tool_id, 0)
            failure_info = f" (Failures: {failure_count})" if failure_count > 0 else ""

            # Display tool with potential failure count
            st.sidebar.markdown(f"**{tool['name']}**{failure_info}: {tool['description']}")

            # Show the sequence of tools used
            if 'tool_sequence' in tool:
                sequence_str = " → ".join([step['tool'] for step in tool['tool_sequence']])
                st.sidebar.markdown(f"*Sequence: {sequence_str}*")