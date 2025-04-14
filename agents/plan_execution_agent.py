# math_solver/agents/plan_execution_agent.py
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional, Tuple
from utils.logging_utils import setup_logger
from core.math_toolbox import MathToolbox
from utils.exceptions import StopException
import re

logger = setup_logger("plan_execution_agent")


class PlanExecutionAgent:
    """Agent that executes the steps from a solution plan to solve math problems."""

    def __init__(self, toolbox: MathToolbox):
        self.toolbox = toolbox
        self.execution_history = []
        self.re = re

        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

        # Create a mapping from tool names to their functions
        self.tool_mapping = {
            "sum": self.toolbox.sum,
            "product": self.toolbox.product,
            "divide": self.toolbox.divide,
            "subtract": self.toolbox.subtract,
            "power": self.toolbox.power,
            "sqrt": self.toolbox.sqrt,
            "modulo": self.toolbox.modulo,
            "round_number": self.toolbox.round_number,
            "avg": self.toolbox.avg
        }

    def execute_plan(self, problem: str, plan: Dict[str, Any], callback_handler=None) -> Dict[str, Any]:
        """
        Execute a solution plan for a given math problem.

        Args:
            problem: The math problem to solve
            plan: The structured plan created by the planner agent
            callback_handler: Optional callback handler for streaming output

        Returns:
            Dictionary containing the execution results
        """
        # Reset execution history
        self.execution_history = []

        if callback_handler:
            callback_handler.container.markdown("## ⚙️ Executing Solution Plan")
            callback_handler.container.write("Plan Execution Agent is working through the plan steps...")

        try:
            # Initialize variables to track the execution
            step_results = {}
            intermediate_results = []

            # Execute each step in the plan
            # In math_solver/agents/plan_execution_agent.py
            # Update part of the execute_plan method

            # Inside the execute_plan method, update the part that processes steps:
            for idx, step in enumerate(plan.get("steps", [])):
                step_num = idx + 1
                description = step.get("description", "")
                tool_info = step.get("tool_info", {})

                if callback_handler:
                    callback_handler.container.markdown(f"### Step {step_num}: {description}")

                # Check if we have tool information
                if tool_info and "tool" in tool_info:
                    tool_name = tool_info["tool"]
                    raw_input = tool_info["input"]

                    # Process the input - replace references to previous steps with actual values
                    try:
                        processed_input = self._process_tool_input(raw_input, step_results)

                        if callback_handler:
                            callback_handler.container.write(f"Using tool: {tool_name} with input: {processed_input}")

                        # Execute the tool
                        if tool_name in self.tool_mapping:
                            tool_func = self.tool_mapping[tool_name]
                            result = tool_func(processed_input)

                            # Track tool usage in execution history
                            self.execution_history.append({
                                "tool": tool_name,
                                "tool_input": processed_input
                            })

                            # Store the result
                            step_results[step_num] = result

                            if callback_handler:
                                callback_handler.container.write(f"Result: {result}")

                            intermediate_results.append({
                                "step": step_num,
                                "description": description,
                                "tool": tool_name,
                                "input": processed_input,
                                "result": result
                            })
                        else:
                            error_msg = f"Unknown tool: {tool_name}"
                            if callback_handler:
                                callback_handler.container.error(error_msg)
                            logger.error(error_msg)

                            intermediate_results.append({
                                "step": step_num,
                                "description": description,
                                "tool": tool_name,
                                "input": processed_input,
                                "error": error_msg
                            })

                    except Exception as e:
                        error_msg = f"Error executing step {step_num}: {str(e)}"
                        if callback_handler:
                            callback_handler.container.error(error_msg)
                        logger.error(error_msg)

                        # Still track the error in the history
                        intermediate_results.append({
                            "step": step_num,
                            "description": description,
                            "tool": tool_name if 'tool_name' in locals() else "unknown",
                            "input": raw_input,
                            "error": str(e)
                        })

                        # We might want to continue despite errors in some cases
                        # For now, we'll continue and let the agent try to recover
                else:
                    # If no tool info, just record the step description
                    if callback_handler:
                        callback_handler.container.write(f"No tool specified for this step. Continuing to next step.")

                    intermediate_results.append({
                        "step": step_num,
                        "description": description,
                        "note": "No tool execution for this step"
                    })

            # Process the final step to determine the result
            final_step = plan.get("final_step", {})
            final_description = final_step.get("description", "")
            result_info = final_step.get("result_info", "")

            if callback_handler:
                callback_handler.container.markdown(f"### Final Step: {final_description}")
                callback_handler.container.write(f"Result information: {result_info}")

            # Determine the final result - typically the last step result
            final_result = None
            if step_results:
                final_step_num = max(step_results.keys())
                final_result = step_results[final_step_num]

                if callback_handler:
                    callback_handler.container.success(f"Final result: {final_result}")
            else:
                final_result = "Could not determine final result - no valid step results"
                if callback_handler:
                    callback_handler.container.error(final_result)

            # Format the solution text
            solution_text = self._format_solution(problem, final_result, result_info)

            if callback_handler:
                callback_handler.container.markdown("## 🎯 Solution")
                callback_handler.container.markdown(solution_text)

            return {
                "problem": problem,
                "solution": solution_text,
                "final_result": final_result,
                "execution_history": self.execution_history,
                "intermediate_results": intermediate_results,
                "successful": final_result is not None and "error" not in str(final_result).lower()
            }

        except StopException:
            # Propagate stop exceptions (e.g., waiting for user input)
            raise
        except Exception as e:
            logger.error(f"Error executing plan: {str(e)}")
            if callback_handler:
                callback_handler.container.error(f"Execution error: {str(e)}")

            return {
                "problem": problem,
                "error": str(e),
                "solution": f"Failed to execute plan: {str(e)}",
                "execution_history": self.execution_history,
                "successful": False
            }

    def _process_tool_input(self, raw_input: str, step_results: Dict[int, str]) -> str:
        """
        Process the tool input string by replacing references to previous step results.

        Args:
            raw_input: The raw input string from the plan
            step_results: Dictionary of results from previous steps

        Returns:
            Processed input string
        """
        processed_input = raw_input

        # Fix for list-style inputs: [80, 80] -> "80, 80"
        list_pattern = r'\[([^\]]+)\]'
        list_match = self.re.search(list_pattern, processed_input)
        if list_match:
            # Extract the content inside brackets and keep it without the brackets
            content = list_match.group(1)
            # Replace any unnecessary quotes
            content = content.replace("'", "").replace('"', "")
            # Replace the entire bracketed expression with just the content
            processed_input = self.re.sub(list_pattern, content, processed_input)
            logger.info(f"Converted list input: {raw_input} -> {processed_input}")

        # Check for references to previous steps like "result from step 1"
        import re

        # Pattern for various ways to reference previous step results
        patterns = [
            r"result (?:from|of) step (\d+)",
            r"step (\d+) result",
            r"result (\d+)",
            r"value (?:from|of) step (\d+)",
            r"output (\d+)"  # Added this pattern to match [OUTPUT 1]
        ]

        for pattern in patterns:
            matches = re.findall(pattern, processed_input, re.IGNORECASE)
            for match in matches:
                step_num = int(match)
                if step_num in step_results:
                    # Replace the reference with the actual result
                    replace_pattern = f"result from step {step_num}"
                    processed_input = processed_input.replace(replace_pattern, str(step_results[step_num]))

                    replace_pattern = f"step {step_num} result"
                    processed_input = processed_input.replace(replace_pattern, str(step_results[step_num]))

                    replace_pattern = f"result {step_num}"
                    processed_input = processed_input.replace(replace_pattern, str(step_results[step_num]))

                    replace_pattern = f"value from step {step_num}"
                    processed_input = processed_input.replace(replace_pattern, str(step_results[step_num]))

                    replace_pattern = f"OUTPUT {step_num}"
                    processed_input = processed_input.replace(replace_pattern, str(step_results[step_num]))

                    replace_pattern = f"output {step_num}"
                    processed_input = processed_input.replace(replace_pattern, str(step_results[step_num]))

        logger.info(f"Processed input: {raw_input} -> {processed_input}")
        return processed_input

    def _format_solution(self, problem: str, final_result: str, result_info: str) -> str:
        """
        Format the final solution text.

        Args:
            problem: The original problem
            final_result: The final calculation result
            result_info: Additional information about the result from the plan

        Returns:
            Formatted solution text
        """
        # Use LLM to generate a natural language solution
        prompt = f"""
        Given this math problem: "{problem}"

        The calculated final result is: {final_result}

        Additional context about the result: {result_info}

        Please provide a clear, concise solution statement that:
        1. Acknowledges the original problem
        2. States the final answer clearly
        3. Uses proper mathematical terminology
        4. Is direct and to the point

        Format your response to start directly with "The answer to..."
        """

        try:
            response = self.llm.invoke(prompt)
            solution_text = response.content.strip()
            return solution_text
        except Exception as e:
            logger.error(f"Error formatting solution: {str(e)}")
            # Fallback to simple format if LLM fails
            return f"The answer to the problem is {final_result}."