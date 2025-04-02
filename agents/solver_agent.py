# math_solver/agents/solver_agent.py
# math_solver/agents/solver_agent.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Any, Optional
from config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from utils.logging_utils import setup_logger
from core.math_toolbox import MathToolbox
from core.virtual_tool_manager import VirtualToolManager
from utils.exceptions import StopException

logger = setup_logger("solver_agent")


class MathSolverAgent:
    """Agent that solves math problems by using available tools."""

    def __init__(self, toolbox: MathToolbox, virtual_tool_manager: VirtualToolManager):
        self.toolbox = toolbox
        self.virtual_tool_manager = virtual_tool_manager
        self.execution_history = []

        # Initialize LLM
        # - temp=0 to reduce hallucinations
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

        # Create basic tools
        self.base_tools = [
            Tool(
                name="sum",
                func=self.toolbox.sum,
                description="Add a list of numbers. Format: 'num1, num2, num3, ...'"
            ),
            Tool(
                name="product",
                func=self.toolbox.product,
                description="Multiply a list of numbers. Format: 'num1, num2, num3, ...'"
            ),
            Tool(
                name="divide",
                func=self.toolbox.divide,
                description="Divide first number by second number. Format: 'num1, num2'"
            ),
            Tool(
                name="subtract",
                func=self.toolbox.subtract,
                description="Subtract second number from first number. Format: 'num1, num2'"
            ),
            Tool(
                name="power",
                func=self.toolbox.power,
                description="Raise first number to the power of second number. Format: 'base, exponent'"
            ),
            Tool(
                name="sqrt",
                func=self.toolbox.sqrt,
                description="Calculate the square root of a number."
            ),
            Tool(
                name="modulo",
                func=self.toolbox.modulo,
                description="Calculate the remainder when first number is divided by second. Format: 'num1, num2'"
            ),
            Tool(
                name="round_number",
                func=self.toolbox.round_number,
                description="Round a number to the specified decimal places. Format: 'number, decimal_places'"
            ),
            Tool(
                name="get_user_input",
                func=self.streamlit_user_input,
                description="Get input from the user by asking a question."
            )
        ]

        # Initialize agent
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agent = initialize_agent(
            tools=self.base_tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def streamlit_user_input(self, question: str) -> str:
        """
        Get input from the user using Streamlit's interactive components.
        Designed to handle inputs within a math workflow.

        Args:
            question: The question to ask the user

        Returns:
            str: The user's input
        """
        from utils.exceptions import StopException

        # Create a unique key for this question
        key_base = f"user_input_{hash(question) % 10000}"
        input_key = f"{key_base}_field"
        submit_key = f"{key_base}_submit"

        # Create a container with a prominent style to draw attention
        st.markdown("""
        <style>
        .user-input-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #4CAF50;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='user-input-container'>", unsafe_allow_html=True)
        st.markdown("### ⚠️ User Input Required")
        st.info(f"**Question from Math Agent**: {question}")

        # Check if we already have an answer
        if key_base in st.session_state:
            answer = st.session_state[key_base]
            st.success(f"Your response: {answer}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Clean up the session state
            del st.session_state[key_base]
            return answer

        # If not, show the input form
        user_input = st.text_input("Your answer:", key=input_key)
        submit_button = st.button("Submit Answer", key=submit_key)

        if submit_button and user_input:
            # Store the answer in session state
            st.session_state[key_base] = user_input
            st.success("Response received! Processing your answer...")
            st.markdown("</div>", unsafe_allow_html=True)

            # Important: Store in a different session state variable to track that we've submitted
            st.session_state[f"{key_base}_submitted"] = True

            # Force a rerun to continue with the updated state
            st.rerun()

        # Add a message to make it clear what's happening
        st.warning("Please enter your answer and click 'Submit Answer' to continue.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Add a timeout mechanism to prevent infinite loops
        if f"{key_base}_timeout_counter" not in st.session_state:
            st.session_state[f"{key_base}_timeout_counter"] = 0

        # Increment the timeout counter
        st.session_state[f"{key_base}_timeout_counter"] += 1

        # If we've been waiting too long (e.g., 10 reruns), provide default value
        if st.session_state[f"{key_base}_timeout_counter"] > 10:
            st.error("No input received after several attempts. Using default value.")
            del st.session_state[f"{key_base}_timeout_counter"]
            return "0"  # Safe default

        # Pause execution to wait for user input
        raise StopException("Waiting for user input")


    def solve_problem(self, problem: str, callback_handler=None) -> str:
        """Solve a math problem by using available tools and virtual tools."""
        # Check if this is sqrt of negative number
        is_sqrt_negative = "square root" in problem.lower() and any(
            term in problem for term in ["-1", "negative", "minus"]
        )

        if is_sqrt_negative:
            logger.info("Detected square root of negative number problem")
            # We may want to handle this directly to avoid repeated tool calls

        # Check if we have a virtual tool for this type of problem
        virtual_tool = self.virtual_tool_manager.find_matching_virtual_tool(problem)

        if virtual_tool:
            # When you see this amount of logging, you know something happened
            # Let's log what's going on with our virtual_tools that we create
            logger.info(f"Using virtual tool: Tool Name : {virtual_tool['name']}")
            logger.info(f"-----> virtual tool {virtual_tool['name']}: {virtual_tool}")
            logger.info(f"-----> virtual tool {virtual_tool['name']}: {problem}, {self.toolbox}")
            fn_ptr = virtual_tool["function"]
            try:
                # there's an issue here, the function isn't returning a value
                result = fn_ptr(input_str=problem, math_toolbox=self.toolbox)
                logger.info(f"-----> virtual tool {virtual_tool['name']} result: {result}")
                return f"Solved using virtual tool {virtual_tool['name']}: {result}"
            except Exception as e:
                logger.warning(f"Could not reuse the function {e}")
                # If virtual tool fails, fall back to regular solving

        # Reset execution history for this problem
        self.execution_history = []

        # Create a wrapped version of each tool that will track usage
        wrapped_tools = []
        for tool in self.base_tools:
            original_func = tool.func

            # Create a wrapper function that logs tool usage
            def make_wrapper(tool_name, original_func):
                def wrapped_func(tool_input):
                    # Log the tool usage
                    self.execution_history.append({
                        "tool": tool_name,
                        "tool_input": tool_input
                    })
                    logger.info(f"Tool used: {tool_name} with input: {tool_input}")
                    # Call the original function
                    return original_func(tool_input)

                return wrapped_func

            # Create a new tool with the wrapped function
            wrapped_tool = Tool(
                name=tool.name,
                func=make_wrapper(tool.name, original_func),
                description=tool.description
            )
            wrapped_tools.append(wrapped_tool)

        # Create a temporary agent with the wrapped tools
        temp_agent = initialize_agent(
            tools=wrapped_tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

        # Combine callbacks
        callbacks = []
        if callback_handler:
            callbacks.append(callback_handler)

        # Use the agent to solve the problem
        try:
            # For sqrt of negative numbers, provide specific guidance
            if is_sqrt_negative:
                additional_guidance = (
                    " For square roots of negative numbers, you need to express the result as an imaginary number. "
                    "Use the sqrt tool with the negative number directly, it will handle complex numbers correctly."
                )
            else:
                additional_guidance = ""

            result = temp_agent.invoke(
                {
                    "input": "Start with a fresh new math question with no priors."
                             f"Solve this math problem: {problem}. Use the available tools. "
                             "Do NOT perform calculations yourself. Always use a tool for ANY numerical calculation. "
                             f"{additional_guidance}"
                },
                callbacks=callbacks
            )

            # Check if result makes sense and doesn't contain errors
            if "error" not in result["output"].lower():
                # Log the execution history to verify it's being populated
                logger.info(f"Execution history: {self.execution_history}")

                # For sqrt of negative numbers, if we have many repeated calls, optimize the sequence
                if is_sqrt_negative and len(self.execution_history) > 3:
                    # Check if all calls are sqrt(-1)
                    all_sqrt_negative = all(
                        step["tool"] == "sqrt" and step["tool_input"].strip() == "-1"
                        for step in self.execution_history
                    )

                    if all_sqrt_negative:
                        logger.info("Optimizing execution history for sqrt of negative number")
                        # Keep just one call
                        self.execution_history = [self.execution_history[0]]

                if self.execution_history:  # Only record if we have tool usage
                    # Record successful sequence for future use
                    self.virtual_tool_manager.record_successful_sequence(
                        problem=problem,
                        sequence=self.execution_history,
                        result=result["output"]
                    )

                    # Check if we can now use the virtual tool for this problem
                    problem_hash = self.virtual_tool_manager.hash_problem(problem)
                    if problem_hash in self.virtual_tool_manager.virtual_tools:
                        logger.info(f"Created new virtual tool for problem: {problem}")
                else:
                    logger.warning(f"No tool usage recorded for problem: {problem}")

                return result["output"]
        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}")
            return f"Failed to solve the problem: {str(e)}"

