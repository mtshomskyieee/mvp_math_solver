# math_solver/agents/math_planner_agent.py
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional, Tuple
from utils.logging_utils import setup_logger
from config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE

logger = setup_logger("math_planner_agent")


class MathPlannerAgent:
    """Agent that plans the steps to solve a math problem."""

    def __init__(self):
        # Initialize LLM with streaming disabled by default
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

    def plan_solution(self, problem: str, toolbox_description: str, callback_handler=None) -> Dict[str, Any]:
        """
        Plan the solution steps for a given math problem.

        Args:
            problem: The math problem to solve
            toolbox_description: Description of available tools
            callback_handler: Optional callback handler for streaming output

        Returns:
            Dictionary containing the plan details
        """
        if callback_handler:
            callback_handler.container.markdown("## 📝 Planning Solution Steps")
            callback_handler.container.write("Math Planner Agent is analyzing the problem...")

        try:
            # Craft a prompt to generate a step-by-step plan
            prompt = f"""
            You are a mathematical planning expert who creates precise step-by-step plans to solve math problems.
            Given a problem, create a detailed plan that outlines exactly what operations to perform in what sequence.

            Available tools: {toolbox_description}

            For the problem: "{problem}"

            Create a step-by-step plan with the following information:
            1. Identify the type of problem and the operations required
            2. Break down the solution into clear, sequential steps 
            3. For each step, specify which tool should be used with what inputs
            4. Be precise about how the outputs from one step feed into subsequent steps

            Format your response in this structure:
            PROBLEM_TYPE: [Type of mathematical problem]

            STEP 1: [Description] -> Use tool: [tool_name] with input: [specific_input]
            STEP 2: [Description] -> Use tool: [tool_name] with input: [specific_input] 
            ...

            FINAL_STEP: [Description] -> Final answer will be: [how to interpret final result]
            """

            # Create a fresh LLM instance with the appropriate streaming setting
            # This helps avoid callback conflicts
            streaming_llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                streaming=callback_handler is not None
            )

            response = streaming_llm.invoke(prompt)

            plan_text = response.content.strip()

            if callback_handler:
                callback_handler.container.markdown("### 📋 Solution Plan")
                callback_handler.container.markdown(plan_text)

            # Parse the plan text into structured steps
            plan = self._parse_plan(plan_text)

            return {
                "problem": problem,
                "plan_text": plan_text,
                "structured_plan": plan
            }

        except Exception as e:
            logger.error(f"Error in planning solution: {str(e)}")
            if callback_handler:
                callback_handler.container.error(f"Planning error: {str(e)}")
            return {
                "problem": problem,
                "error": str(e),
                "plan_text": "Failed to create plan due to an error."
            }

    def _parse_plan(self, plan_text: str) -> Dict[str, Any]:
        """
        Parse the plan text into a structured format.

        Args:
            plan_text: The raw plan text

        Returns:
            Structured plan dictionary
        """
        lines = plan_text.split('\n')

        problem_type = ""
        steps = []
        final_step = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to extract problem type
            if line.startswith("PROBLEM_TYPE:"):
                problem_type = line[len("PROBLEM_TYPE:"):].strip()

            # Try to extract steps
            elif line.startswith("STEP "):
                try:
                    # Extract step number
                    step_num_end = line.find(":")
                    if step_num_end > 0:
                        # Extract description and tool info
                        desc_tool_split = line.find("->")
                        if desc_tool_split > 0:
                            description = line[step_num_end + 1:desc_tool_split].strip()
                            tool_info_text = line[desc_tool_split + 2:].strip()

                            # Extract tool and input
                            tool_match = tool_info_text.find("with input:")
                            if tool_match > 0:
                                tool_name = tool_info_text[len("Use tool:"):tool_match].strip()
                                tool_input = tool_info_text[tool_match + len("with input:"):].strip()

                                # Clean up the tool name - REMOVE ANY BRACKETS
                                tool_name = tool_name.strip('[]').strip()

                                # Clean up the tool input format
                                # Remove any unnecessary quotes
                                tool_input = tool_input.strip('"\'[]')

                                steps.append({
                                    "description": description,
                                    "tool_info": {
                                        "tool": tool_name,
                                        "input": tool_input
                                    }
                                })
                            else:
                                # If no tool specified, just add the description
                                steps.append({"description": description})
                        else:
                            # If no tool info, just add the description
                            steps.append({"description": line[step_num_end + 1:].strip()})
                except Exception as e:
                    logger.error(f"Error parsing step: {str(e)}")
                    # Add what we can
                    steps.append({"description": line})

            # Try to extract final step
            elif line.startswith("FINAL_STEP:"):
                final_text = line[len("FINAL_STEP:"):].strip()
                # Try to extract description and result info
                result_split = final_text.find("Final answer will be:")
                if result_split > 0:
                    description = final_text[:result_split].strip()
                    result_info = final_text[result_split + len("Final answer will be:"):].strip()

                    # Clean up any brackets in the result info
                    result_info = result_info.strip('[]')

                    final_step = {
                        "description": description,
                        "result_info": result_info
                    }
                else:
                    final_step = {"description": final_text}

        return {
            "problem_type": problem_type,
            "steps": steps,
            "final_step": final_step
        }