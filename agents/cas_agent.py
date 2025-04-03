# math_solver/agents/cas_agent.py
import re
import sympy
from typing import Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from utils.logging_utils import setup_logger

logger = setup_logger("cas_agent")


class CASAgent:
    """Agent that uses Computer Algebra System (SymPy) to solve math problems."""

    def __init__(self):
        # Initialize LLM for framing/interpreting problems
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

    def _extract_numeric_value(self, text: str):
        """Extract numeric value from text."""
        # First check for complex numbers (like i, 3i, etc.) with a more specific pattern
        # Make sure it only matches when there's literally an 'i' character
        complex_match = re.search(r'([-+]?\d*\.?\d*)\s*i\b', text)
        if complex_match:
            coefficient = complex_match.group(1)
            # Handle empty coefficient (just "i")
            if coefficient == '':
                return complex(0, 1)
            # Handle just "+" (like "+i")
            elif coefficient == '+':
                return complex(0, 1)
            # Handle just "-" (like "-i")
            elif coefficient == '-':
                return complex(0, -1)
            # Handle numeric coefficients
            else:
                try:
                    # Convert to float and create complex number
                    coef_float = float(coefficient)
                    return complex(0, coef_float)
                except ValueError:
                    # If conversion fails for some reason, default to 1
                    return complex(0, 1)

        # Then try to extract regular numeric values
        numeric_match = re.search(r'[-+]?\d*\.?\d+', text)
        if numeric_match:
            return float(numeric_match.group(0))

        return None
    def _parse_equation(self, problem: str):
        """Parse a mathematical equation from the problem text."""
        # Use LLM to convert word problem to symbolic equation
        prompt = f"""
        Convert the following mathematical problem to a SymPy-compatible expression:
        Problem: {problem}

        Return ONLY the expression, nothing else. For example:
        - For "What is 25 + 37?" return "25 + 37"
        - For "square root of -1" return "sqrt(-1)"
        - For "round 1.578" return "round(1.578)"
        - For "(5 mod 2)" return "5 % 2"
        - For "(2-3)*5^2" return "(2-3)*(5**2)"

        Expression:
        """

        response = self.llm.invoke(prompt)
        expression = response.content.strip()
        logger.info(f"Parsed expression: {expression}")
        return expression

    def _solve_with_sympy(self, expression: str):
        """Solve the equation using SymPy."""
        try:
            # Replace common operations with SymPy syntax
            expression = expression.replace("^", "**")

            # Handle modulo operation
            if "mod" in expression:
                expression = expression.replace("mod", "%")

            # Handle square root of negative numbers
            if "sqrt(-" in expression:
                # Use sympy's I for imaginary unit
                expression = expression.replace("sqrt(-", "sqrt(")
                # Evaluate with complex numbers enabled
                result = complex(sympy.sympify(expression, evaluate=True))
                if result.real == 0:
                    return f"{result.imag}i" if result.imag != 1 else "i"
                else:
                    return f"{result.real} + {result.imag}i" if result.imag > 0 else f"{result.real} - {abs(result.imag)}i"

            # Handle rounding
            if "round" in expression:
                num = re.search(r'round\s*\(\s*([-+]?\d*\.?\d+)\s*\)', expression)
                if num:
                    return str(round(float(num.group(1))))

            # Evaluate the expression
            result = sympy.sympify(expression, evaluate=True)

            # Convert result to a readable form
            if isinstance(result, sympy.Integer):
                return str(int(result))
            elif isinstance(result, sympy.Float):
                return str(float(result))
            elif isinstance(result, sympy.Rational):
                # Convert fraction to decimal
                return str(float(result))
            elif isinstance(result, sympy.core.numbers.ComplexInfinity):
                return "undefined (division by zero)"
            elif sympy.I in result.atoms():
                # It's a complex number
                return str(complex(result))
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Error solving with SymPy: {str(e)}")
            return f"Error: {str(e)}"

    def solve_problem(self, problem: str, callback_handler=None) -> str:
        """Solve a math problem using Computer Algebra System."""
        try:
            if callback_handler:
                callback_handler.container.markdown("## 🧮 CAS Agent Analysis")
                callback_handler.container.write("CAS Agent is parsing the problem...")

            # Parse the equation
            expression = self._parse_equation(problem)

            if callback_handler:
                callback_handler.container.write(f"Parsed expression: `{expression}`")
                callback_handler.container.write("Solving with SymPy...")

            # Solve with SymPy
            result = self._solve_with_sympy(expression)

            if callback_handler:
                callback_handler.container.markdown(f"**CAS result:** `{result}`")
                # Make the final result more prominent
                callback_handler.container.success(f"CAS computed: {result}")

            return result

        except Exception as e:
            logger.error(f"Error in CAS agent: {str(e)}")
            if callback_handler:
                callback_handler.container.error(f"CAS agent error: {str(e)}")
            return f"Failed to solve: {str(e)}"

    def verify_result(self, problem: str, proposed_solution: str, callback_handler=None) -> Tuple[bool, str]:
        """Verify a result by comparing against CAS solution."""
        try:
            if callback_handler:
                callback_handler.container.markdown("## 🔍 CAS Verification")

            # Get CAS solution
            cas_solution = self.solve_problem(problem, callback_handler)

            if callback_handler:
                callback_handler.container.markdown(f"**Verifying solution:**")
                callback_handler.container.markdown(f"- Proposed: `{proposed_solution}`")
                callback_handler.container.markdown(f"- CAS: `{cas_solution}`")

            # Extract numeric values for comparison
            proposed_num = self._extract_numeric_value(proposed_solution)
            cas_num = self._extract_numeric_value(cas_solution)

            # If both have numeric values, compare them
            if proposed_num is not None and cas_num is not None:
                # Use a small tolerance for floating point comparisons
                tolerance = 0.001 * max(abs(float(cas_num.real if hasattr(cas_num, 'real') else cas_num)), 1.0)

                # Compare real and imaginary parts for complex numbers
                if isinstance(proposed_num, complex) or isinstance(cas_num, complex):
                    proposed_complex = complex(proposed_num) if not isinstance(proposed_num, complex) else proposed_num
                    cas_complex = complex(cas_num) if not isinstance(cas_num, complex) else cas_num

                    real_match = abs(proposed_complex.real - cas_complex.real) <= tolerance
                    imag_match = abs(proposed_complex.imag - cas_complex.imag) <= tolerance

                    is_verified = real_match and imag_match
                else:
                    # For real numbers
                    is_verified = abs(float(proposed_num) - float(cas_num)) <= tolerance

                verification_msg = f"CAS verification: {'VERIFIED' if is_verified else 'INCORRECT'} - Expected {cas_solution}"
                return is_verified, verification_msg

            # If we couldn't extract numeric values, check if the strings are similar
            else:
                # Normalize strings for comparison
                proposed_clean = proposed_solution.lower().replace(" ", "")
                cas_clean = cas_solution.lower().replace(" ", "")

                # Check for specific patterns like "i" or complex numbers
                is_verified = False

                # Both indicate complex/imaginary number
                if ("i" in proposed_clean and "i" in cas_clean) or (
                        "imaginary" in proposed_clean and "imaginary" in cas_clean):
                    is_verified = True
                # Both indicate undefined or infinity
                elif ("undefined" in proposed_clean and "undefined" in cas_clean) or (
                        "infinity" in proposed_clean and "infinity" in cas_clean):
                    is_verified = True
                # Direct string comparison
                elif proposed_clean == cas_clean:
                    is_verified = True

                if is_verified:
                    if callback_handler:
                        callback_handler.container.success(
                            f"✅ CAS Verification: MATCH! {proposed_solution} matches expected {cas_solution}")
                else:
                    if callback_handler:
                        callback_handler.container.error(
                            f"❌ CAS Verification: MISMATCH! {proposed_solution} doesn't match expected {cas_solution}")

                verification_msg = f"CAS verification: {'VERIFIED' if is_verified else 'INCORRECT'} - Expected {cas_solution}"
                return is_verified, verification_msg

        except Exception as e:
            logger.error(f"Error in CAS verification: {str(e)}")
            return False, f"CAS verification error: {str(e)}"