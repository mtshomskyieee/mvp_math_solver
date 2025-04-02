# math_solver/core/virtual_tool_manager.py
import csv
import hashlib
import inspect
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from utils.logging_utils import setup_logger
from core.math_toolbox import MathToolbox

logger = setup_logger("virtual_tool_manager")

class VirtualToolManager:
    """Manages virtual tools created from successful sequences of tool calls."""

    def __init__(self, max_failures=3):
        self.virtual_tools = {}  # Maps problem_hash -> virtual tool
        self.successful_sequences = {}  # Maps problem_hash -> sequence of tool calls
        self.tool_failure_counts = {}  # Maps problem_hash -> failure count
        self.max_failures = max_failures  # Maximum allowed failures before removing a tool

    def hash_problem(self, problem: str) -> str:
        """Create a hash for a problem to identify similar problems."""
        # Extract key features from the problem for similarity matching
        # For simplicity, we're just using a hash of the problem text
        return hashlib.md5(problem.lower().strip().encode()).hexdigest()

    def record_successful_sequence(self, problem: str, sequence: List[Dict[str, Any]], result: str):
        """Record a successful sequence of tool calls for a problem."""
        problem_hash = self.hash_problem(problem)
        logger.info(f"storing successful sequence: {sequence}")
        self.successful_sequences[problem_hash] = {
            "problem": problem,
            "sequence": sequence,
            "result": result,
            "created_at": datetime.now().isoformat()
        }

        # Create a virtual tool
        self._create_virtual_tool(problem_hash)

    def _optimize_tool_sequence(self, tool_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize a tool sequence by removing redundant repeated calls.

        This is particularly useful for cases like sqrt(-1) where the agent
        might repeatedly call the same tool with the same input.
        """
        if not tool_sequence:
            return []

        # Special case handling for sqrt of negative numbers
        # If there are any sqrt calls with negative numbers, we want to keep just one
        has_sqrt_negative = any(
            step["tool"] == "sqrt" and float(step["tool_input"]) < 0
            for step in tool_sequence
            if step["tool"] == "sqrt" and step["tool_input"].strip().replace('-', '').replace('.', '').isdigit()
        )

        if has_sqrt_negative:
            # Find the first sqrt call with a negative number
            for step in tool_sequence:
                if step["tool"] == "sqrt" and step["tool_input"].strip().startswith('-'):
                    # Return just this step
                    logger.info(f"Optimized sqrt of negative number sequence to a single call")
                    return [step]

        # General case: optimize any sequence with repeating identical steps
        optimized = []
        last_step = None
        repeat_count = 0

        for step in tool_sequence:
            if last_step and step["tool"] == last_step["tool"] and step["tool_input"] == last_step["tool_input"]:
                # Same tool with same input as previous step
                repeat_count += 1
                # Limit to at most 2 repetitions
                if repeat_count <= 2:
                    optimized.append(step)
            else:
                # Different tool or input
                optimized.append(step)
                last_step = step
                repeat_count = 0

        return optimized

    def _create_virtual_tool(self, problem_hash: str):
        """Create a virtual tool from a successful sequence."""
        sequence_data = self.successful_sequences[problem_hash]
        problem_type = self._categorize_problem(sequence_data["problem"])

        # Get the sequence of tools used to solve this problem
        tool_sequence = sequence_data["sequence"]
        original_problem = sequence_data["problem"]

        # OPTIMIZATION: Check for repeated identical tool calls and compress them
        if len(tool_sequence) > 3:
            optimized_sequence = self._optimize_tool_sequence(tool_sequence)
            if len(optimized_sequence) < len(tool_sequence):
                logger.info(f"Optimized tool sequence from {len(tool_sequence)} to {len(optimized_sequence)} steps")
                tool_sequence = optimized_sequence
                # Update the sequence in the successful_sequences storage as well
                self.successful_sequences[problem_hash]["sequence"] = tool_sequence

        # Determine the primary tool for naming/description purposes
        primary_tool = None
        if len(tool_sequence) > 0:
            primary_tool = tool_sequence[0]["tool"]

        # Check if this is a special case pattern like (a-b)*c^d
        pattern_match = re.search(r'\((\d+)-(\d+)\)\*(\d+)\^(\d+)', original_problem)
        is_special_pattern = pattern_match is not None

        # Check for specific patterns that need special handling
        is_simple_multiplication = re.search(r'multiply\s+(\d+)\s+by\s+(\d+)', original_problem.lower()) is not None
        is_sqrt_negative = False
        if "square root" in original_problem.lower() and "-" in original_problem:
            is_sqrt_negative = True
            logger.info(f"Detected special case: square root of negative number")

        # Pre-process original problem context for faster lookups later
        original_numbers_with_context = self._parse_expression(original_problem)

        # Extract the specific values for simple multiplication pattern if present
        mult_first_num = None
        mult_second_num = None
        if is_simple_multiplication:
            mult_match = re.search(r'multiply\s+(\d+)\s+by\s+(\d+)', original_problem.lower())
            if mult_match:
                mult_first_num = mult_match.group(1)
                mult_second_num = mult_match.group(2)
                logger.info(f"Detected simple multiplication: {mult_first_num} × {mult_second_num}")

        # Create a function that will execute the entire sequence of tools
        def virtual_tool_func(input_str: str, math_toolbox: MathToolbox) -> str:
            try:
                # Create a cache key that includes the tool and input
                cache_key = f"{problem_hash}:{input_str}"

                # Check global class-level cache first (faster than per-tool cache)
                if not hasattr(self, '_global_cache'):
                    self._global_cache = {}

                # Fast cache lookup
                if cache_key in self._global_cache:
                    logger.info(f"Global cache hit for input: {input_str}")
                    return self._global_cache[cache_key]

                # SPECIAL CASE: Handle simple multiplication pattern
                if is_simple_multiplication:
                    # Try to extract the numbers from the new problem
                    new_mult_match = re.search(r'multiply\s+(\d+)\s+by\s+(\d+)', input_str.lower())
                    if new_mult_match:
                        new_first_num = new_mult_match.group(1)
                        new_second_num = new_mult_match.group(2)
                        logger.info(f"Detected simple multiplication in input: {new_first_num} × {new_second_num}")

                        # Directly use these values instead of trying to map them
                        try:
                            result = math_toolbox.product(f"{new_first_num}, {new_second_num}")
                            logger.info(f"Direct multiplication result: {result}")

                            # Cache and return the result
                            self._global_cache[cache_key] = result
                            return result
                        except Exception as e:
                            logger.error(f"Error in direct multiplication: {e}")
                            # Fall through to the regular processing if direct approach fails

                # Special case for sqrt of negative number
                if is_sqrt_negative:
                    # Extract the number from the input
                    num_match = re.search(r'-?\d+\.?\d*', input_str)
                    if num_match:
                        number = float(num_match.group(0))
                        if number < 0:
                            # Directly calculate sqrt of negative number
                            result = math_toolbox.sqrt(str(number))
                            self._global_cache[cache_key] = result
                            return result

                # Early pattern detection for special cases to avoid expensive calculations
                if is_special_pattern:
                    new_pattern_match = re.search(r'\((\d+)-(\d+)\)\*(\d+)\^(\d+)', input_str)
                    if new_pattern_match:
                        a, b, c, d = [new_pattern_match.group(i) for i in range(1, 5)]

                        # Fast path for special pattern - avoid logging for speed
                        try:
                            step1_result = math_toolbox.subtract(f"{a}, {b}")
                            if "Error" in step1_result:
                                return f"Error in calculation: {step1_result}"

                            step2_result = math_toolbox.power(f"{c}, {d}")
                            if "Error" in step2_result:
                                return f"Error in calculation: {step2_result}"

                            final_result = math_toolbox.product(f"{step1_result}, {step2_result}")

                            # Cache result globally
                            self._global_cache[cache_key] = final_result
                            return final_result
                        except Exception as e:
                            return f"Error: {str(e)}"

                # Only parse the expression if we need to (expensive operation)
                numbers_with_context = self._parse_expression(input_str)

                # Log the numbers found in the input for debugging
                logger.info(f"Numbers in input: {[n['value'] for n in numbers_with_context]}")
                logger.info(f"Original numbers: {[n['value'] for n in original_numbers_with_context]}")

                # Force a different mapping approach for specific patterns
                force_position_mapping = False

                # For multiplication problems, force position-based mapping
                if "multiply" in input_str.lower() and "multiply" in original_problem.lower():
                    force_position_mapping = True
                    logger.info("Forcing position-based mapping for multiplication problem")

                # Map numbers from input to original problem
                if force_position_mapping or (
                        len(numbers_with_context) == len(original_numbers_with_context) and
                        len(numbers_with_context) <= 4):
                    # Use position-based mapping for simple problems with the same number of values
                    number_mapping = {}
                    for i in range(min(len(numbers_with_context), len(original_numbers_with_context))):
                        orig_val = original_numbers_with_context[i]['value']
                        new_val = numbers_with_context[i]['value']
                        number_mapping[orig_val] = new_val
                    logger.info(f"Using position-based mapping: {number_mapping}")
                else:
                    # Use the more sophisticated mapping for complex problems
                    number_mapping = self._map_numbers(numbers_with_context, original_numbers_with_context)
                    logger.info(f"Using sophisticated mapping: {number_mapping}")

                # Initialize results dictionary with pre-allocated size
                step_results = {}

                # Execute each tool in the sequence with minimal overhead
                for step_idx, step in enumerate(tool_sequence):
                    tool_name = step["tool"]
                    original_tool_input = step["tool_input"]
                    input_values = [x.strip() for x in original_tool_input.split(',')]
                    formatted_inputs = []

                    # Process each input value with minimal branching
                    for i, input_val in enumerate(input_values):
                        # Fast path for previous results
                        used_previous = False
                        for prev_idx in range(step_idx):
                            if prev_idx in step_results:
                                prev_result = step_results[prev_idx]
                                try:
                                    # Fast numeric comparison
                                    if isinstance(prev_result, (int, float)) and abs(
                                            float(input_val) - prev_result) < 0.0001:
                                        formatted_inputs.append(str(prev_result))
                                        used_previous = True
                                        break
                                    # String comparison only if needed
                                    elif str(prev_result) == input_val:
                                        formatted_inputs.append(str(prev_result))
                                        used_previous = True
                                        break
                                except ValueError:
                                    pass

                        if not used_previous:
                            # Use mapped value or original
                            if input_val in number_mapping:
                                mapped_val = number_mapping[input_val]
                                formatted_inputs.append(mapped_val)
                                logger.info(f"Mapped {input_val} to {mapped_val}")
                            else:
                                formatted_inputs.append(input_val)
                                logger.info(f"Using unmapped value: {input_val}")

                    # Format input string
                    formatted_input = ", ".join(formatted_inputs)
                    logger.info(f"Step {step_idx + 1}: {tool_name}({formatted_input})")

                    # Fast tool dispatch using dictionary instead of if/elif chain
                    tool_dispatch = {
                        "sum": math_toolbox.sum,
                        "product": math_toolbox.product,
                        "divide": math_toolbox.divide,
                        "subtract": math_toolbox.subtract,
                        "power": math_toolbox.power,
                        "sqrt": math_toolbox.sqrt,
                        "modulo": math_toolbox.modulo,
                        "round_number": math_toolbox.round_number
                    }

                    try:
                        if tool_name not in tool_dispatch:
                            raise ValueError(f"Unknown tool: {tool_name}")

                        result = tool_dispatch[tool_name](formatted_input)
                        logger.info(f"Step {step_idx + 1} result: {result}")

                        # Check for errors
                        if "Error occurred:" in result:
                            return f"Virtual tool execution failed at step {step_idx + 1}: {result}"

                        # Store result efficiently
                        try:
                            step_results[step_idx] = float(result)
                        except ValueError:
                            step_results[step_idx] = result

                    except Exception as e:
                        error_msg = f"Step {step_idx + 1} failed: {str(e)}"
                        logger.error(error_msg)
                        return error_msg

                # Get final result
                final_result = str(step_results.get(len(tool_sequence) - 1, "No result"))
                logger.info(f"Final result: {final_result}")

                # Cache result globally for future use
                self._global_cache[cache_key] = final_result
                return final_result

            except Exception as e:
                logger.error(f"Virtual tool execution failed: {str(e)}")
                return f"Virtual tool execution failed: {str(e)}"

        # Register the virtual tool
        self.virtual_tools[problem_hash] = {
            "name": f"VirtualTool_{problem_type}_{problem_hash[:8]}",
            "description": f"Solves {problem_type} problems similar to: '{sequence_data['problem']}'",
            "function": virtual_tool_func,
            "primary_tool": primary_tool,
            "tool_sequence": tool_sequence
        }

        logger.info(f"Created virtual tool '{self.virtual_tools[problem_hash]['name']}' "
                    f"using sequence of {len(tool_sequence)} tools: {[step['tool'] for step in tool_sequence]}")

    def _parse_expression(self, expression: str) -> List[Dict[str, Any]]:
        """
        Parse a mathematical expression to extract numbers with their context.
        Returns a list of dictionaries with value, position, and context.
        """
        # Extract numbers with their positions
        numbers = []
        for match in re.finditer(r'-?\d+\.?\d*', expression):
            value = match.group(0)
            position = match.start()

            # Determine context - look at characters before and after
            left_context = expression[:position].strip()
            right_context = expression[position + len(value):].strip()

            # Check for operations and parentheses
            left_paren = '(' in left_context and ')' not in left_context[:left_context.rfind('(')]
            right_paren = ')' in right_context and '(' not in right_context[:right_context.find(')')]

            # Check for operators near the number
            left_op = None
            right_op = None

            for op in ['+', '-', '*', '/', '^']:
                if left_context and left_context[-1] == op:
                    left_op = op
                if right_context and right_context[0] == op:
                    right_op = op

            numbers.append({
                'value': value,
                'position': position,
                'left_paren': left_paren,
                'right_paren': right_paren,
                'left_op': left_op,
                'right_op': right_op
            })

        return numbers

    def _map_numbers(self, new_numbers: List[Dict[str, Any]], original_numbers: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Map numbers from the new problem to corresponding positions in the original problem.
        Uses context information to match numbers better than just position.
        """
        mapping = {}

        # Track which new numbers have been mapped
        used_new_numbers = set()

        # Identify if we're dealing with a complex expression like (a-b)*c^d
        is_complex_expr = (len(original_numbers) >= 4 and
                           any(n['right_op'] == '^' for n in original_numbers) and
                           any(n['left_op'] == '*' for n in original_numbers))

        # First step: Map based on structural position for certain expressions
        if is_complex_expr and len(new_numbers) >= 4:
            # For expressions like (a-b)*c^d, map by position and operators
            for i, orig_num in enumerate(original_numbers):
                if i < len(new_numbers):
                    # Check for matching operators which indicate equivalent structure
                    new_num = new_numbers[i]
                    if (orig_num['left_op'] == new_num['left_op'] and
                            orig_num['right_op'] == new_num['right_op']):
                        mapping[orig_num['value']] = new_num['value']
                        used_new_numbers.add(i)
                    # Special handling for power operations (base ^ exponent)
                    elif orig_num['right_op'] == '^' and new_num['right_op'] == '^':
                        mapping[orig_num['value']] = new_num['value']
                        used_new_numbers.add(i)

        # Second step: Map remaining numbers based on context similarity
        for i, orig_num in enumerate(original_numbers):
            if orig_num['value'] in mapping:
                continue  # Already mapped

            best_match = None
            best_score = -1

            for j, new_num in enumerate(new_numbers):
                if j in used_new_numbers:
                    continue

                # Calculate context similarity score
                score = 0

                # Match parentheses
                if orig_num['left_paren'] == new_num['left_paren']:
                    score += 1
                if orig_num['right_paren'] == new_num['right_paren']:
                    score += 1

                # Match operators (very important)
                if orig_num['left_op'] == new_num['left_op']:
                    score += 3
                if orig_num['right_op'] == new_num['right_op']:
                    score += 3

                # Position similarity
                position_diff = abs(i - j)
                if position_diff == 0:
                    score += 2
                elif position_diff == 1:
                    score += 1

                if score > best_score:
                    best_score = score
                    best_match = j

            # If we found a good match, add it to mapping
            if best_match is not None and best_score >= 2:
                mapping[orig_num['value']] = new_numbers[best_match]['value']
                used_new_numbers.add(best_match)

        # Special case handling for common expression patterns
        if is_complex_expr and len(new_numbers) >= 4:
            # For expressions like (a-b)*c^d, ensure the exponent is mapped correctly
            # Find the power operation and make sure exponent is correct
            for i, orig_num in enumerate(original_numbers):
                if orig_num['right_op'] == '^' and i + 1 < len(original_numbers):
                    # This is the base of a power operation, next number is the exponent
                    orig_base = orig_num['value']
                    orig_exp = original_numbers[i + 1]['value']

                    # If base is already mapped but exponent isn't, add it
                    if orig_base in mapping and orig_exp not in mapping and i + 1 < len(new_numbers):
                        mapping[orig_exp] = new_numbers[i + 1]['value']

        logger.info(f"Final number mapping: {mapping}")
        return mapping
    def _categorize_problem(self, problem: str) -> str:
        """Categorize the type of math problem."""
        # This is a simple categorization - in a real system this would be more sophisticated
        if "add" in problem.lower() or "sum" in problem.lower() or "plus" in problem.lower():
            return "Addition"
        elif "multiply" in problem.lower() or "product" in problem.lower():
            return "Multiplication"
        elif "divide" in problem.lower() or "quotient" in problem.lower():
            return "Division"
        elif "subtract" in problem.lower() or "difference" in problem.lower() or "minus" in problem.lower():
            return "Subtraction"
        elif "power" in problem.lower() or "exponent" in problem.lower() or "square" in problem.lower():
            return "Exponentiation"
        elif "symbolic" in problem.lower() or "derivative" in problem.lower() or "integral" in problem.lower():
            return "Calculus"
        else:
            return "General"

    def record_tool_failure(self, problem_hash: str) -> bool:
        """
        Record a failure for a virtual tool and remove it if it exceeds the maximum allowed failures.

        Args:
            problem_hash: The hash of the problem associated with the failing tool

        Returns:
            bool: True if the tool was removed, False otherwise
        """
        if problem_hash not in self.tool_failure_counts:
            self.tool_failure_counts[problem_hash] = 0

        self.tool_failure_counts[problem_hash] += 1

        # Check if we should remove this tool
        if self.tool_failure_counts[problem_hash] >= self.max_failures:
            if problem_hash in self.virtual_tools:
                tool_name = self.virtual_tools[problem_hash]['name']
                logger.warning(f"Removing unreliable virtual tool '{tool_name}' after {self.max_failures} failures")

                # Remove the tool
                del self.virtual_tools[problem_hash]

                # Optionally, also remove the successful sequence to prevent recreating the same tool
                if problem_hash in self.successful_sequences:
                    del self.successful_sequences[problem_hash]

                # Clear the failure count
                del self.tool_failure_counts[problem_hash]

                return True

        return False

    def _identify_primary_operation(self, problem: str) -> str:
        """Identify the primary operation type of a problem."""
        problem_lower = problem.lower()

        # Check for modulo operation first (since it might contain division-like terms)
        if 'mod' in problem_lower or '%' in problem_lower or 'modulo' in problem_lower or 'remainder' in problem_lower:
            return "modulo"

        # Check for other operations
        if any(term in problem_lower for term in ['add', 'sum', 'plus', '+']):
            return "addition"
        if any(term in problem_lower for term in ['subtract', 'minus', 'difference', '-']):
            return "subtraction"
        if any(term in problem_lower for term in ['multiply', 'product', 'times', '*']):
            return "multiplication"
        if any(term in problem_lower for term in ['divide', 'quotient', '/']):
            return "division"
        if any(term in problem_lower for term in ['power', 'exponent', '^', 'squared', 'cubed']):
            return "exponentiation"
        if 'sqrt' in problem_lower or 'square root' in problem_lower:
            return "square_root"
        if 'round' in problem_lower:
            return "rounding"

        # Default to "unknown" if no clear operation is identified
        return "unknown"

    # Add this method to the VirtualToolManager class

    def _is_tool_relevant_for_problem(self, problem: str, tool_sequence: List[Dict[str, Any]]) -> bool:
        """
        Determine if a tool sequence is relevant for a given problem.
        This helps prevent misapplication of complex sequences to simple problems.
        """
        # Extract numbers from the problem to count operands
        problem_numbers = self._parse_expression(problem)

        # Check for operand count mismatch in addition problems
        if "+" in problem:
            # Count the number of addition operators to determine how many numbers we're adding
            plus_count = problem.count("+")
            expected_num_count = plus_count + 1  # N operators means N+1 operands

            # If the problem has a different number of operands than what the tool expects
            if len(problem_numbers) != expected_num_count:
                logger.info(
                    f"Tool sequence not relevant: problem has {len(problem_numbers)} numbers, needs {expected_num_count}")
                return False

            # For addition problems, check if the tool actually uses the sum tool
            has_sum_tool = any(step["tool"] == "sum" for step in tool_sequence)
            if not has_sum_tool:
                logger.info(f"Tool sequence not relevant: addition problem but no sum tool used")
                return False

        # Similar check for multiplication
        if "*" in problem or "×" in problem or "multiply" in problem.lower():
            # Check if there's a product tool in the sequence for multiplication problems
            has_product_tool = any(step["tool"] == "product" for step in tool_sequence)
            if not has_product_tool:
                logger.info(f"Tool sequence not relevant: multiplication problem but no product tool used")
                return False

        # Check for simple multiplication problems
        is_simple_multiplication = re.search(r'multiply\s+(\d+)\s+by\s+(\d+)', problem.lower()) is not None
        if is_simple_multiplication:
            # For simple multiplication, don't use sequences with round_number
            for step in tool_sequence:
                if step["tool"] == "round_number":
                    return False

        # Check the number of tools used against the complexity of the problem
        # For each operator in the problem, we typically need at least one tool
        operator_count = sum(problem.count(op) for op in ['+', '-', '*', '/', '^'])
        if len(tool_sequence) < operator_count:
            logger.info(
                f"Tool sequence not relevant: problem needs at least {operator_count} operations but tool has {len(tool_sequence)}")
            return False

        # By default, consider the tool relevant
        return True


    def find_matching_virtual_tool(self, problem: str) -> Optional[Dict[str, Any]]:
        """Find a virtual tool that can solve a similar problem."""
        problem_hash = self.hash_problem(problem)
        problem_lower = problem.lower()

        # First, identify the primary operation type
        operation_type = self._identify_primary_operation(problem)
        logger.info(f"Identified primary operation: {operation_type}")

        # Check for exact match
        if problem_hash in self.virtual_tools:
            tool = self.virtual_tools[problem_hash]
            # Make sure the tool sequence is appropriate for this problem
            if 'tool_sequence' in tool and self._is_tool_relevant_for_problem(problem, tool['tool_sequence']):
                return tool
            else:
                logger.info(f"Found exact match tool {tool['name']} but it's not relevant for this problem type.")
                return None

        # Parse the current problem to extract its structure
        new_problem_numbers = self._parse_expression(problem)

        # Count expected operations based on numbers in the problem
        expected_op_count = self._estimate_operation_count(problem, new_problem_numbers)
        logger.info(f"Estimated operations needed for problem: {expected_op_count}")

        best_match = None
        best_match_score = 0

        # Consider all available tools
        for hash_key, tool in self.virtual_tools.items():
            if 'tool_sequence' not in tool:
                continue

            # Get the original problem associated with this tool
            original_problem = None
            if hash_key in self.successful_sequences:
                original_problem = self.successful_sequences[hash_key]["problem"]
            else:
                continue

            # Check if the operation types match
            original_operation = self._identify_primary_operation(original_problem)
            if original_operation != operation_type:
                logger.info(
                    f"Tool {tool['name']} has primary operation {original_operation}, but problem needs {operation_type}")
                continue

            # Skip tools with too many or too few operations
            tool_op_count = len(tool['tool_sequence'])
            if abs(tool_op_count - expected_op_count) > 1:  # Allow small variations
                logger.info(
                    f"Tool {tool['name']} has {tool_op_count} operations, but problem needs ~{expected_op_count}")
                continue

            # Extract numbers from the original problem
            original_problem_numbers = self._parse_expression(original_problem)

            # Compare the number of operands in both problems
            # For operations like addition and multiplication, the number of operands must match
            if len(new_problem_numbers) != len(original_problem_numbers):
                logger.info(
                    f"Tool {tool['name']} works with {len(original_problem_numbers)} numbers, but problem has {len(new_problem_numbers)} numbers")
                continue

            # Check operation types
            if not self._has_compatible_operations(problem, tool['tool_sequence']):
                logger.info(f"Tool {tool['name']} has incompatible operation types for this problem")
                continue

            # Now check if the tool is relevant for this problem using our improved method
            if not self._is_tool_relevant_for_problem(problem, tool['tool_sequence']):
                logger.info(f"Tool {tool['name']} is not relevant for this problem type")
                continue

            # Get structure similarity score
            structure_score = self._calculate_structure_similarity(
                new_problem_numbers,
                original_problem_numbers
            )

            # Calculate overall match score
            problem_tokens = set(problem.lower().split())
            description = tool["description"].lower()
            desc_tokens = set(description.split())
            token_overlap = len(problem_tokens.intersection(desc_tokens))

            match_score = structure_score * 0.9 + min(1.0, token_overlap / 5) * 0.3

            if match_score > best_match_score and match_score > 0.6:  # Threshold for a good match
                best_match = tool
                best_match_score = match_score

        if best_match:
            logger.info(f"Selected tool {best_match['name']} with match score {best_match_score:.2f}")
            return best_match

        return None
    def _estimate_operation_count(self, problem: str, numbers: List[Dict[str, Any]]) -> int:
        """Estimate how many operations are needed based on the problem text and numbers."""
        # Count explicit operation words
        op_count = 0

        # Count based on operation keywords
        operations = {
            'add': re.findall(r'\b(?:add|plus|sum)\b', problem.lower()),
            'subtract': re.findall(r'\b(?:subtract|minus|difference)\b', problem.lower()),
            'multiply': re.findall(r'\b(?:multiply|times|product)\b', problem.lower()),
            'divide': re.findall(r'\b(?:divide|division|quotient)\b', problem.lower()),
        }

        for op_type, matches in operations.items():
            op_count += len(matches)

        # Count based on mathematical symbols
        op_symbols = ['+', '-', '*', '/', '^']
        for symbol in op_symbols:
            op_count += problem.count(symbol)

        # If we have n numbers, we typically need n-1 operations
        # This is a fallback if we couldn't detect operations explicitly
        if op_count == 0 and len(numbers) > 1:
            op_count = len(numbers) - 1

        # Ensure at least 1 operation if we have numbers
        return max(1, op_count) if numbers else 0

    def _has_compatible_operations(self, problem: str, tool_sequence: List[Dict[str, Any]]) -> bool:
        """Check if the tool sequence has compatible operation types for the problem."""
        # Extract operation types from the problem
        problem_lower = problem.lower()

        # Check for specific operation types in the problem
        has_addition = any(term in problem_lower for term in ['add', 'sum', 'plus', '+'])
        has_subtraction = any(term in problem_lower for term in ['subtract', 'minus', 'difference', '-'])
        has_multiplication = any(term in problem_lower for term in ['multiply', 'product', 'times', '*'])
        has_division = any(term in problem_lower for term in ['divide', 'quotient', '/'])
        has_power = any(term in problem_lower for term in ['power', 'exponent', '^', 'squared', 'cubed'])
        has_square_root = 'sqrt' in problem_lower or 'square root' in problem_lower
        # Add specific check for modulo operations
        has_modulo = 'mod' in problem_lower or '%' in problem_lower or 'modulo' in problem_lower or 'remainder' in problem_lower

        # Count operation types in the tool sequence
        tool_operations = {
            'sum': 0,
            'subtract': 0,
            'product': 0,
            'divide': 0,
            'power': 0,
            'sqrt': 0,
            'modulo': 0,
            'round_number': 0
        }

        for step in tool_sequence:
            if step['tool'] in tool_operations:
                tool_operations[step['tool']] += 1

        # Check for compatibility based on operations
        if has_addition and tool_operations['sum'] == 0:
            logger.info("Problem requires addition but tool has no sum operation")
            return False
        if has_subtraction and tool_operations['subtract'] == 0:
            logger.info("Problem requires subtraction but tool has no subtract operation")
            return False
        if has_multiplication and tool_operations['product'] == 0:
            logger.info("Problem requires multiplication but tool has no product operation")
            return False
        if has_division and tool_operations['divide'] == 0 and not has_modulo:
            logger.info("Problem requires division but tool has no divide operation")
            return False
        if has_power and tool_operations['power'] == 0:
            logger.info("Problem requires exponentiation but tool has no power operation")
            return False
        if has_square_root and tool_operations['sqrt'] == 0:
            logger.info("Problem requires square root but tool has no sqrt operation")
            return False
        # Add specific check for modulo operations
        if has_modulo and tool_operations['modulo'] == 0:
            logger.info("Problem requires modulo but tool has no modulo operation")
            return False

        # Make sure we don't misuse division tools for modulo operations and vice versa
        if has_modulo and not has_division and tool_operations['divide'] > 0 and tool_operations['modulo'] == 0:
            logger.info("Problem requires modulo but tool only has division")
            return False
        if has_division and not has_modulo and tool_operations['modulo'] > 0 and tool_operations['divide'] == 0:
            logger.info("Problem requires division but tool only has modulo")
            return False

        # For subtraction specifically, check if the number of operations matches
        if has_subtraction:
            # Count subtraction operations in problem
            subtract_count = problem_lower.count('subtract') + problem_lower.count('minus') + problem_lower.count('-')
            # If problem has clearly more subtractions than the tool can handle
            if subtract_count > tool_operations['subtract'] + 1:  # Allow some flexibility
                logger.info(
                    f"Problem needs ~{subtract_count} subtractions but tool only has {tool_operations['subtract']}")
                return False

        return True

    def _is_tool_relevant_for_problem(self, problem: str, tool_sequence: List[Dict[str, Any]]) -> bool:
        """
        Determine if a tool sequence is relevant for a given problem.
        This helps prevent misapplication of complex sequences to simple problems.
        """
        # Extract numbers from the problem to count operands
        problem_numbers = self._parse_expression(problem)
        problem_lower = problem.lower()

        # Check for specific operations in the problem text
        has_modulo = 'mod' in problem_lower or '%' in problem_lower or 'modulo' in problem_lower or 'remainder' in problem_lower
        has_division = any(term in problem_lower for term in ['divide', 'quotient', '/']) and not has_modulo

        # Specific check for modulo operations
        if has_modulo:
            # Check if the tool sequence has a modulo operation
            has_modulo_tool = any(step["tool"] == "modulo" for step in tool_sequence)
            if not has_modulo_tool:
                logger.info(f"Tool sequence not relevant: modulo problem but no modulo tool used")
                return False

            # Check operand count for modulo operations
            if len(problem_numbers) != 2:  # Modulo needs exactly 2 operands
                logger.info(f"Tool sequence not relevant: modulo needs 2 numbers but found {len(problem_numbers)}")
                return False

        # Check for division vs modulo mismatch
        if has_division:
            has_divide_tool = any(step["tool"] == "divide" for step in tool_sequence)
            if not has_divide_tool:
                logger.info(f"Tool sequence not relevant: division problem but no divide tool used")
                return False

        # Check for operand count mismatch in addition problems
        if "+" in problem:
            # Count the number of addition operators to determine how many numbers we're adding
            plus_count = problem.count("+")
            expected_num_count = plus_count + 1  # N operators means N+1 operands

            # If the problem has a different number of operands than what the tool expects
            if len(problem_numbers) != expected_num_count:
                logger.info(
                    f"Tool sequence not relevant: problem has {len(problem_numbers)} numbers, needs {expected_num_count}")
                return False

            # For addition problems, check if the tool actually uses the sum tool
            has_sum_tool = any(step["tool"] == "sum" for step in tool_sequence)
            if not has_sum_tool:
                logger.info(f"Tool sequence not relevant: addition problem but no sum tool used")
                return False

        # Similar check for multiplication
        if "*" in problem or "×" in problem or "multiply" in problem.lower():
            # Check if there's a product tool in the sequence for multiplication problems
            has_product_tool = any(step["tool"] == "product" for step in tool_sequence)
            if not has_product_tool:
                logger.info(f"Tool sequence not relevant: multiplication problem but no product tool used")
                return False

        # Check for simple multiplication problems
        is_simple_multiplication = re.search(r'multiply\s+(\d+)\s+by\s+(\d+)', problem.lower()) is not None
        if is_simple_multiplication:
            # For simple multiplication, don't use sequences with round_number
            for step in tool_sequence:
                if step["tool"] == "round_number":
                    return False

        # Check the number of tools used against the complexity of the problem
        # For each operator in the problem, we typically need at least one tool
        operator_count = sum(problem.count(op) for op in ['+', '-', '*', '/', '^', '%'])
        if len(tool_sequence) < operator_count:
            logger.info(
                f"Tool sequence not relevant: problem needs at least {operator_count} operations but tool has {len(tool_sequence)}")
            return False

        # By default, consider the tool relevant
        return True

    def _calculate_structure_similarity(self, new_numbers: List[Dict[str, Any]],
                                        original_numbers: List[Dict[str, Any]]) -> float:
        """
        Calculate a similarity score between 0 and 1 for the structure of two problems
        based on their extracted numbers and context.
        """
        # If number counts are very different, it's not a good match
        if abs(len(new_numbers) - len(original_numbers)) > 1:
            return 0.2  # Low base similarity

        # If both have no numbers, return medium similarity
        if len(new_numbers) == 0 and len(original_numbers) == 0:
            return 0.5

        # Calculate similarity based on patterns of numbers and operations
        similarity = 0.0

        # Start with base similarity based on number count match
        if len(new_numbers) == len(original_numbers):
            similarity += 0.5
        else:
            similarity += 0.3

        # Check pattern of operations between numbers
        operation_pattern_match = self._compare_operation_patterns(new_numbers, original_numbers)
        similarity += operation_pattern_match * 0.5

        return min(1.0, similarity)  # Cap at 1.0

    def _compare_operation_patterns(self, nums1: List[Dict[str, Any]],
                                    nums2: List[Dict[str, Any]]) -> float:
        """
        Compare the pattern of operations between two sequences of numbers.
        Returns a score between 0 and 1.
        """
        if not nums1 or not nums2:
            return 0.0

        # Extract operation patterns
        def get_ops_pattern(nums):
            pattern = []
            for i in range(len(nums) - 1):
                # Check operations between adjacent numbers
                curr = nums[i]
                next_num = nums[i + 1]

                # Use left_op of next number or right_op of current
                op = next_num.get('left_op') or curr.get('right_op')
                pattern.append(op)
            return pattern

        pattern1 = get_ops_pattern(nums1)
        pattern2 = get_ops_pattern(nums2)

        # If patterns are different lengths, take the shorter one
        min_len = min(len(pattern1), len(pattern2))
        if min_len == 0:
            return 0.0

        # Count matching operations
        matches = sum(1 for i in range(min_len) if pattern1[i] == pattern2[i])
        return matches / min_len

    # Add these imports at the top of virtual_tool_manager.py
    import csv
    import os
    import inspect
    import datetime

    def serialize_virtual_tool(self, problem_hash: str) -> str:
        """
        Serialize a virtual tool to a string representation.

        Args:
            problem_hash: The hash of the problem associated with the virtual tool

        Returns:
            String representation of the virtual tool function
        """
        if problem_hash not in self.virtual_tools:
            return "Virtual tool not found"

        virtual_tool = self.virtual_tools[problem_hash]

        # Get the function source code
        #func_src = inspect.getsource(virtual_tool["function"])

        # Extract sequence information for inclusion in the serialized output
        sequence_info = []
        if "tool_sequence" in virtual_tool:
            for step in virtual_tool["tool_sequence"]:
                sequence_info.append(f"{step['tool']}('{step['tool_input']}')")

        # Create a formatted string representation
        tool_str = f"""
    # Virtual Tool: {virtual_tool['name']}
    # Description: {virtual_tool['description']}
    # Created: {datetime.now().isoformat()}
    # Sequence: {' -> '.join(sequence_info)}
    #
    """
        return tool_str

    def save_virtual_tools_to_csv(self, filename="new_tools.csv"):
        """
        Save all successful virtual tools to a CSV file.

        Args:
            filename: Name of the CSV file to save to
        """
        # Check if we have any virtual tools to save
        if not self.virtual_tools:
            logger.info("No virtual tools to save")
            return

        # Determine if we're creating a new file or appending
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ["problem_text", "output", "virtual_tool_as_string"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if creating new file
            if not file_exists:
                writer.writeheader()

            # Write each tool to the CSV
            tools_saved = 0
            for problem_hash, tool in self.virtual_tools.items():
                # Skip tools without successful sequences
                if problem_hash not in self.successful_sequences:
                    continue

                # Get the problem and result
                sequence_data = self.successful_sequences[problem_hash]
                problem_text = sequence_data["problem"]
                output = sequence_data["result"]

                # Serialize the tool
                tool_str = self.serialize_virtual_tool(problem_hash)

                # Write to CSV
                writer.writerow({
                    "problem_text": problem_text,
                    "output": output,
                    "virtual_tool_as_string": tool_str
                })
                tools_saved += 1

            logger.info(f"Saved {tools_saved} virtual tools to {filename}")

    # Add this method to the VirtualToolManager class in math_solver/core/virtual_tool_manager.py

    def import_virtual_tools_from_csv(self, filename="new_tools.csv"):
        """
        Import virtual tools from a CSV file and create virtual tool functions.

        Args:
            filename: Name of the CSV file to read from
        """
        import csv
        import re

        if not os.path.exists(filename):
            logger.warning(f"CSV file {filename} not found.")
            return 0

        tools_imported = 0

        try:
            with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)  # Use csv.reader instead of DictReader

                # Skip header row
                headers = next(reader, None)

                for row in reader:
                    try:
                        # Ensure we have at least 3 columns
                        if len(row) < 3:
                            logger.warning(f"Skipping row: Not enough columns. Found {len(row)}")
                            continue

                        # Extract data from columns
                        problem_text = row[0]
                        output = row[1]
                        tool_info = row[2]

                        # Ensure all fields are strings
                        problem_text = str(problem_text) if problem_text else ""
                        output = str(output) if output else ""
                        tool_info = str(tool_info) if tool_info else ""

                        if not problem_text:
                            logger.warning("Skipping row: No problem text found")
                            continue

                        # Hash the problem to create a consistent ID
                        problem_hash = self.hash_problem(problem_text)

                        # Skip if we already have this tool
                        if problem_hash in self.virtual_tools:
                            logger.info(f"Skipping tool for problem '{problem_text}': Already exists")
                            continue

                        # Clean up the tool info
                        tool_info = self._clean_tool_info(tool_info)

                        # Log for debugging
                        logger.info(f"Processing problem: '{problem_text}'")
                        logger.info(f"Tool info: '{tool_info[:100]}...'")

                        # Extract tool name and description using regex
                        name_match = re.search(r'# Virtual Tool: (.*?)$', tool_info, re.MULTILINE)
                        desc_match = re.search(r'# Description: (.*?)$', tool_info, re.MULTILINE)
                        sequence_match = re.search(r'# Sequence: (.*?)$', tool_info, re.MULTILINE)

                        if not (name_match and desc_match):
                            logger.warning("Skipping row: Could not extract tool name or description")
                            continue

                        tool_name = name_match.group(1).strip()
                        tool_description = desc_match.group(1).strip()

                        # Extract and parse the sequence
                        sequence_str = sequence_match.group(1).strip() if sequence_match else ""
                        logger.info(f"Sequence string: '{sequence_str}'")

                        tool_sequence = []

                        # Parse the sequence format like: "product('13, 10')"
                        for step in sequence_str.split(" -> "):
                            # Handle the arrow in different formats
                            step = step.strip()
                            match = re.match(r"(\w+)\('([^']*?)'\)", step)
                            if match:
                                tool_name = match.group(1)
                                tool_input = match.group(2)
                                tool_sequence.append({
                                    "tool": tool_name,
                                    "tool_input": tool_input
                                })

                        # Skip if we couldn't parse a valid sequence
                        if not tool_sequence:
                            logger.warning(f"Skipping row: Could not parse sequence from '{sequence_str}'")
                            continue

                        # Log the extracted sequence
                        logger.info(f"Extracted sequence: {tool_sequence}")

                        # Store this as a successful sequence
                        self.successful_sequences[problem_hash] = {
                            "problem": problem_text,
                            "sequence": tool_sequence,
                            "result": output,
                            "created_at": datetime.now().isoformat()
                        }

                        # Create a virtual tool from this sequence
                        self._create_virtual_tool(problem_hash)

                        # Increment counter if successful
                        if problem_hash in self.virtual_tools:
                            tools_imported += 1
                            logger.info(f"Successfully imported tool for problem: '{problem_text}'")
                        else:
                            logger.warning(f"Failed to create virtual tool for problem: '{problem_text}'")

                    except Exception as e:
                        logger.error(f"Error importing tool: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())  # Print the full traceback
                        continue

            logger.info(f"Imported {tools_imported} virtual tools from {filename}")
            return tools_imported

        except Exception as e:
            logger.error(f"Failed to import tools from CSV: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # Print the full traceback
            return 0

    def _clean_tool_info(self, tool_info):
        """Clean up the tool info string by removing unnecessary quotes and whitespace."""
        # Ensure we have a string
        if not isinstance(tool_info, str):
            if isinstance(tool_info, list):
                # Convert list to string
                tool_info = str(tool_info[0]) if tool_info else ""
            else:
                tool_info = str(tool_info) if tool_info else ""

        # Remove outer quotes if present
        if tool_info.startswith('"') and tool_info.endswith('"'):
            tool_info = tool_info[1:-1]

        # Remove escaped quotes
        tool_info = tool_info.replace('\\"', '"')

        # Normalize newlines
        tool_info = tool_info.replace('\\n', '\n')

        return tool_info