# math_solver/core/math_toolbox.py
import random
import numpy as np
from typing import Dict, List, Any
from utils.logging_utils import setup_logger

logger = setup_logger("math_toolbox")


class MathToolbox:
    """A collection of math tools, some of which are intentionally unreliable."""

    def __init__(self):
        # Track which tools are unreliable
        self.unreliable_tools = ["sum", "product"]
        # Track tool usage statistics
        self.tool_stats = {
            "sum": {"calls": 0, "errors": 0},
            "product": {"calls": 0, "errors": 0},
            "divide": {"calls": 0, "errors": 0},
            "subtract": {"calls": 0, "errors": 0},
            "power": {"calls": 0, "errors": 0},
            "sqrt": {"calls": 0, "errors": 0},
            "modulo": {"calls": 0, "errors": 0},
            "round_number": {"calls": 0, "errors": 0},
        }

    def set_all_tools_reliable(self):
        # clear the list
        self.unreliable_tools = []

    def unset_all_tools_reliable(self):
        # add the unreliable tools back in
        self.unreliable_tools = ["sum", "product"]

    def get_tools_string(self) -> str:
        tools_string = ",".join(self.tool_stats.keys())
        return tools_string

    def sum(self, numbers_str: str) -> str:
        """Add a list of numbers. Format: 'num1, num2, num3, ...'"""
        self.tool_stats["sum"]["calls"] += 1

        # Introduce errors 40% of the time
        if random.random() < 0.4 and "sum" in self.unreliable_tools:
            self.tool_stats["sum"]["errors"] += 1
            if random.random() < 0.5:  # Sometimes return wrong answer
                return "Error occurred: Invalid input format"
            else:  # Sometimes return incorrect result
                try:
                    numbers = [float(x.strip()) for x in numbers_str.split(',')]
                    # Return an incorrect sum
                    return str(sum(numbers) + random.randint(1, 10))
                except:
                    return "Error occurred: Could not parse numbers"

        try:
            numbers = [float(x.strip()) for x in numbers_str.split(',')]
            return str(sum(numbers))
        except Exception as e:
            self.tool_stats["sum"]["errors"] += 1
            return f"Error occurred: {str(e)}"

    def product(self, numbers_str: str) -> str:
        """Multiply a list of numbers. Format: 'num1, num2, num3, ...'"""
        self.tool_stats["product"]["calls"] += 1

        # Introduce errors 30% of the time
        if random.random() < 0.3 and "product" in self.unreliable_tools:
            self.tool_stats["product"]["errors"] += 1
            if random.random() < 0.5:  # Sometimes throw error
                return "Error occurred: Invalid input format"
            else:  # Sometimes return incorrect result
                try:
                    numbers = [float(x.strip()) for x in numbers_str.split(',')]
                    # Return an incorrect product
                    result = 1
                    for num in numbers:
                        result *= num
                    return str(result * 1.1)  # Wrong by 10%
                except:
                    return "Error occurred: Could not parse numbers"

        try:
            numbers = [float(x.strip()) for x in numbers_str.split(',')]
            result = 1
            for num in numbers:
                result *= num
            return str(result)
        except Exception as e:
            self.tool_stats["product"]["errors"] += 1
            return f"Error occurred: {str(e)}"

    def divide(self, numbers_str: str) -> str:
        """Divide first number by second number. Format: 'num1, num2'"""
        self.tool_stats["divide"]["calls"] += 1
        try:
            numbers = [float(x.strip()) for x in numbers_str.split(',')]
            if len(numbers) != 2:
                raise ValueError("Exactly two numbers are required for division")
            if numbers[1] == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return str(numbers[0] / numbers[1])
        except Exception as e:
            self.tool_stats["divide"]["errors"] += 1
            return f"Error occurred: {str(e)}"

    def subtract(self, numbers_str: str) -> str:
        """Subtract second number from first number. Format: 'num1, num2'"""
        self.tool_stats["subtract"]["calls"] += 1
        try:
            numbers = [float(x.strip()) for x in numbers_str.split(',')]
            if len(numbers) != 2:
                raise ValueError("Exactly two numbers are required for subtraction")
            return str(numbers[0] - numbers[1])
        except Exception as e:
            self.tool_stats["subtract"]["errors"] += 1
            return f"Error occurred: {str(e)}"

    def power(self, numbers_str: str) -> str:
        """Raise first number to the power of second number. Format: 'base, exponent'"""
        self.tool_stats["power"]["calls"] += 1
        try:
            numbers = [float(x.strip()) for x in numbers_str.split(',')]
            if len(numbers) != 2:
                raise ValueError("Exactly two numbers are required (base, exponent)")
            return str(numbers[0] ** numbers[1])
        except Exception as e:
            self.tool_stats["power"]["errors"] += 1
            return f"Error occurred: {str(e)}"

    def sqrt(self, number_str: str) -> str:
        """Calculate the square root of a number, including complex numbers."""
        self.tool_stats["sqrt"]["calls"] += 1
        try:
            number = float(number_str.strip())

            # Handle negative numbers correctly (using complex numbers)
            if number < 0:
                # Calculate the square root as a complex number
                import cmath
                result = cmath.sqrt(number)

                # Format the complex number for readability
                if result.imag == 1.0:
                    return f"{result.real}+i" if result.real != 0 else "i"
                elif result.imag == -1.0:
                    return f"{result.real}-i" if result.real != 0 else "-i"
                else:
                    if result.real == 0:
                        return f"{result.imag}i"
                    else:
                        sign = "+" if result.imag > 0 else ""
                        return f"{result.real}{sign}{result.imag}i"
            else:
                # For non-negative numbers, use regular square root
                return str(np.sqrt(number))
        except Exception as e:
            self.tool_stats["sqrt"]["errors"] += 1
            return f"Error occurred: {str(e)}"

    def modulo(self, numbers_str: str) -> str:
        """Calculate the remainder when first number is divided by second. Format: 'num1, num2'"""
        self.tool_stats["modulo"]["calls"] += 1
        try:
            numbers = [float(x.strip()) for x in numbers_str.split(',')]
            if len(numbers) != 2:
                raise ValueError("Exactly two numbers are required for modulo")
            if numbers[1] == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return str(numbers[0] % numbers[1])
        except Exception as e:
            self.tool_stats["modulo"]["errors"] += 1
            return f"Error occurred: {str(e)}"

    def round_number(self, input_str: str) -> str:
        """Round a number to the specified decimal places. Format: 'number, decimal_places'"""
        self.tool_stats["round_number"]["calls"] += 1
        try:
            parts = [x.strip() for x in input_str.split(',')]
            if len(parts) != 2:
                raise ValueError("Input should be 'number, decimal_places'")
            number = float(parts[0])
            decimal_places = int(parts[1])
            return str(round(number, decimal_places))
        except Exception as e:
            self.tool_stats["round_number"]["errors"] += 1
            return f"Error occurred: {str(e)}"

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about tool usage and errors."""
        return self.tool_stats
