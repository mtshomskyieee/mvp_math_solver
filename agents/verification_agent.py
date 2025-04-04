# math_solver/agents/verification_agent.py
import re

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Any, Optional, Tuple
from config.settings import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from utils.logging_utils import setup_logger
from core.math_toolbox import MathToolbox

logger = setup_logger("verification_agent")

class VerificationAgent:
    """Agent that verifies the results of math problems by checking for consistency."""

    def __init__(self, toolbox: MathToolbox):
        self.toolbox = toolbox

        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

        # Create tools
        """
        self.tools = [
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
                name="avg",
                func=self.toolbox.avg,
                description="Round a number to the specified decimal places. Format: 'num1, num2, ...'"
            ),
        ]
        """
        self.tools = []
        # Initialize agent
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def verify_result(self, problem: str, proposed_solution: str, callback_handler=None) -> Tuple[bool, str]:
        """Verify a solution by solving the problem independently and comparing results."""
        try:
            # Extract numeric answer from the proposed solution if possible
            proposed_numeric_answer = self._extract_numeric_value(proposed_solution)
            proposed_nonnumeric_answer = self._extract_nonnumeric_value(proposed_solution)

            # Use the agent to verify the problem
            if callback_handler:
                callback_handler.container.write("Verification agent is independently solving the problem...")

            # This prompt tested well
            """
                                "input": f"Solve this math problem precisely: {problem}. "
                                         f"Use the tools to calculate the exact answer. "
                                         f"After solving, respond with only the result labeled as 'EXACT_ANSWER: ' followed by the answer."
            """

            # This prompt had equations as the exact answer, apparently less is more
            """
              "input": f"Solve this math problem precisely: {problem}. "
                             f"Use the appropriate mathematical tools and methods to derive the exact numeric answer. "
                             f"Format responses using Parenthesis, Plus, Minus, Multiply, and Caret"
                             f"If there is a complex answer, create a polynomial with real and complex numbers"
                             f"Always reduce answer to a decimal approximation when possible."
                             f"sqrt(integer) will be represented by a float; for example √4 = 2.0 "
                             f"After solving, respond with only the result labeled as 'EXACT_ANSWER: ' followed by the answer."

            """

            result = self.agent.invoke(
                {
                    "input": f"Solve this math problem precisely: {problem}. "
                             f"Use the tools to calculate the exact answer. "
                             f"After solving, respond with only the result labeled as 'EXACT_ANSWER: ' followed by the answer."
                },
                callbacks=[callback_handler] if callback_handler else None
            )

            verification_output = result["output"]

            # Extract the exact answer from verification output
            exact_answer = self._extract_numeric_value(verification_output)
            exact_nonnumeric_answer = self._extract_nonnumeric_value(verification_output)

            # Special handling for complex numbers and non-numeric answers
            if exact_nonnumeric_answer and proposed_nonnumeric_answer:
                # Check if both contain 'i' or similar indicators of complex numbers
                if ('i' in exact_nonnumeric_answer.lower() and 'i' in proposed_nonnumeric_answer.lower()) or \
                        (
                                'imaginary' in exact_nonnumeric_answer.lower() and 'imaginary' in proposed_nonnumeric_answer.lower()):
                    verification_msg = f"✅ Answer verified: Complex number answer matches. Expected: {exact_nonnumeric_answer}, Got: {proposed_nonnumeric_answer}"
                    if callback_handler:
                        callback_handler.container.write(verification_msg)
                    return True, verification_msg

                # Check for more general string matching for non-numeric answers
                if self._string_similarity(exact_nonnumeric_answer.lower(), proposed_nonnumeric_answer.lower()) > 0.8:
                    verification_msg = f"✅ Answer verified: {proposed_nonnumeric_answer} matches expected {exact_nonnumeric_answer}"
                    if callback_handler:
                        callback_handler.container.write(verification_msg)
                    return True, verification_msg

            # Continue with existing numeric verification logic
            if exact_answer is not None and proposed_numeric_answer is not None:
                # Compare the numeric values with a small tolerance
                # But only for values that should be exact integers for simple calculations
                if self._is_simple_arithmetic_problem(problem):
                    # For simple arithmetic, expect exact integer results
                    is_verified = abs(float(exact_answer) - float(proposed_numeric_answer)) < 0.01
                else:
                    # For more complex calculations, allow a small tolerance
                    tolerance = 0.001 * max(abs(float(exact_answer)), 1.0)
                    is_verified = abs(float(exact_answer) - float(proposed_numeric_answer)) <= tolerance

                if is_verified:
                    verification_msg = f"VERIFIED: The proposed solution gives the correct result {exact_answer}."
                    if callback_handler:
                        callback_handler.container.write(
                            f"✅ Answer verified: {proposed_numeric_answer} matches expected {exact_answer}")
                    return True, verification_msg
                else:
                    verification_msg = f"INCORRECT: The proposed solution gives {proposed_numeric_answer}, but the correct answer is {exact_answer}."
                    if callback_handler:
                        callback_handler.container.write(
                            f"❌ Answer incorrect: {proposed_numeric_answer} does not match expected {exact_answer}")
                    return False, verification_msg
            else:
                # Fallback if we couldn't extract numeric values
                if "VERIFIED" in verification_output or "correct" in verification_output.lower():
                    return True, verification_output
                elif "INCORRECT" in verification_output or "wrong" in verification_output.lower():
                    return False, verification_output
                else:
                    # Ambiguous result, we'll consider it not verified
                    return False, "Could not definitively verify the solution: " + verification_output
        except Exception as e:
            logger.error(f"Error verifying result: {str(e)}")
            return False, f"Failed to verify the result: {str(e)}"
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract a numeric value from text, handling various formats."""
        try:
            # Check for EXACT_ANSWER format
            exact_answer_match = re.search(r'EXACT_ANSWER:\s*(-?\d+\.?\d*)', text)
            if exact_answer_match:
                return float(exact_answer_match.group(1))

            # Look for numbers after common answer indicators
            answer_indicators = [
                r'answer\s*(?:is|:)?\s*(-?\d+\.?\d*)',
                r'result\s*(?:is|:)?\s*(-?\d+\.?\d*)',
                r'equal\s*(?:to|:)?\s*(-?\d+\.?\d*)',
                r'(?:evaluate|evaluates)\s*(?:to|:)?\s*(-?\d+\.?\d*)'
            ]

            for pattern in answer_indicators:
                match = re.search(pattern, text.lower())
                if match:
                    return float(match.group(1))

            # If no matches found with indicators, look for any number
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                # If there are multiple numbers, prefer the last one as it's often the final answer
                return float(numbers[-1])

            return None
        except Exception as e:
            logger.error(f"Error extracting numeric value: {str(e)}")
            return None

    def _extract_nonnumeric_value(self, text: str) -> Optional[str]:
        """Extract a non-numeric value like complex numbers or variables from text."""
        try:
            # Check for EXACT_ANSWER format
            exact_answer_match = re.search(r'EXACT_ANSWER:\s*([^\n]+)', text)
            if exact_answer_match:
                return exact_answer_match.group(1).strip()

            # Look for complex number answers (including just 'i')
            complex_patterns = [
                r'answer\s*(?:is|:)?\s*([^.,;]*i[^.,;]*)',
                r'result\s*(?:is|:)?\s*([^.,;]*i[^.,;]*)',
                r'equal\s*(?:to|:)?\s*([^.,;]*i[^.,;]*)',
                r'(?:evaluate|evaluates)\s*(?:to|:)?\s*([^.,;]*i[^.,;]*)',
                r'(?:value|equals)\s*(?:is|:)?\s*([^.,;]*i[^.,;]*)'
            ]

            for pattern in complex_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    return match.group(1).strip()

            # Check for standalone 'i' or explicit statements about complex numbers
            if re.search(r'\bi\b|\bimaginary\b|complex', text.lower()):
                # Carefully extract the context around i
                match = re.search(r'([^.,;]*(?:\bi\b|\bimaginary unit\b|\bsquare root of -1\b)[^.,;]*)', text.lower())
                if match:
                    return match.group(1).strip()

            return None
        except Exception as e:
            logger.error(f"Error extracting non-numeric value: {str(e)}")
            return None

    def _string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate the similarity between two strings using Levenshtein distance.
        Returns a value between 0 and 1 where 1 means identical strings.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()

    def _is_simple_arithmetic_problem(self, problem: str) -> bool:
        """
        Determine if a problem is simple arithmetic (should have exact integer results).
        """
        # Look for simple arithmetic keywords and operators
        simple_patterns = [
            r'multiply\s+(\d+)\s+by\s+(\d+)',
            r'add\s+(\d+)\s+(?:and|to)\s+(\d+)',
            r'subtract\s+(\d+)\s+from\s+(\d+)',
            r'divide\s+(\d+)\s+by\s+(\d+)'
        ]

        for pattern in simple_patterns:
            if re.search(pattern, problem.lower()):
                return True

        # Also check for simple numeric expressions
        if re.search(r'^\s*\d+\s*[+\-*/]\s*\d+\s*$', problem):
            return True

        return False
