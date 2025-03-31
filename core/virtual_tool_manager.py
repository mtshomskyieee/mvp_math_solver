# math_solver/core/virtual_tool_manager.py
import hashlib
import json
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

    def _create_virtual_tool(self, problem_hash: str):
        """Create a virtual tool from a successful sequence."""
        sequence_data = self.successful_sequences[problem_hash]
        problem_type = self._categorize_problem(sequence_data["problem"])

        # Get the sequence of tools used to solve this problem
        tool_sequence = sequence_data["sequence"]
        original_problem = sequence_data["problem"]

        # Determine the primary tool for naming/description purposes
        primary_tool = None
        if len(tool_sequence) > 0:
            primary_tool = tool_sequence[0]["tool"]

        # Check if this is a special case pattern like (a-b)*c^d
        pattern_match = re.search(r'\((\d+)-(\d+)\)\*(\d+)\^(\d+)', original_problem)
        is_special_pattern = pattern_match is not None

        # Create a function that will execute the entire sequence of tools
        def virtual_tool_func(input_str: str, math_toolbox: MathToolbox) -> str:
            try:
                logger.info(f"Executing virtual tool with tool_sequence: {json.dumps(tool_sequence)}")

                # Special case for the pattern (a-b)*c^d
                if is_special_pattern:
                    # Try to match the same pattern in the new problem
                    new_pattern_match = re.search(r'\((\d+)-(\d+)\)\*(\d+)\^(\d+)', input_str)
                    if new_pattern_match:
                        # Extract values from the new problem
                        a = new_pattern_match.group(1)
                        b = new_pattern_match.group(2)
                        c = new_pattern_match.group(3)
                        d = new_pattern_match.group(4)

                        logger.info(f"Recognized pattern (a-b)*c^d with values: a={a}, b={b}, c={c}, d={d}")

                        # Execute the steps with the extracted values
                        try:
                            step1_result = math_toolbox.subtract(f"{a}, {b}")
                            logger.info(f"Step 1: subtract({a}, {b}) = {step1_result}")

                            step2_result = math_toolbox.power(f"{c}, {d}")
                            logger.info(f"Step 2: power({c}, {d}) = {step2_result}")

                            # Check for errors before continuing
                            if "Error" in step1_result or "Error" in step2_result:
                                return f"Error in calculation: {step1_result} or {step2_result}"

                            final_result = math_toolbox.product(f"{step1_result}, {step2_result}")
                            logger.info(f"Step 3: product({step1_result}, {step2_result}) = {final_result}")

                            return final_result
                        except Exception as e:
                            logger.error(f"Error executing pattern: {str(e)}")
                            return f"Error: {str(e)}"

                # If not a special pattern or pattern matching failed, fall back to the regular approach
                # Extract numbers from the input string while preserving their position and context
                numbers_with_context = self._parse_expression(input_str)
                logger.info(f"Parsed expression: {numbers_with_context}")

                # Extract numbers from the original problem to match patterns
                original_numbers_with_context = self._parse_expression(original_problem)
                logger.info(f"Original problem parsed: {original_numbers_with_context}")

                # For this specific case, apply direct mapping based on the expression structure
                # For (a-b)*c^d pattern, map the first, second, third and fourth values in order
                if len(numbers_with_context) >= 4 and len(original_numbers_with_context) >= 4:
                    number_mapping = {}
                    for i in range(min(len(numbers_with_context), len(original_numbers_with_context))):
                        orig_val = original_numbers_with_context[i]['value']
                        new_val = numbers_with_context[i]['value']
                        number_mapping[orig_val] = new_val

                    logger.info(f"Direct position mapping: {number_mapping}")
                else:
                    # Fall back to the regular mapping approach for non-pattern cases
                    number_mapping = self._map_numbers(numbers_with_context, original_numbers_with_context)
                    logger.info(f"Regular number mapping: {number_mapping}")

                # Initialize intermediate results dictionary
                step_results = {}  # Maps step index to result

                # Execute each tool in the sequence
                for step_idx, step in enumerate(tool_sequence):
                    tool_name = step["tool"]
                    original_tool_input = step["tool_input"]

                    # Parse the original inputs
                    input_values = [x.strip() for x in original_tool_input.split(',')]

                    # Prepare formatted inputs
                    formatted_inputs = []

                    # Process each input value
                    for i, input_val in enumerate(input_values):
                        # First check if this input is a result from a previous step
                        is_previous_result = False
                        for prev_idx in range(step_idx):
                            if prev_idx in step_results:
                                prev_result = step_results[prev_idx]
                                # Compare as string and as float
                                prev_result_str = str(prev_result)
                                try:
                                    # If input can be parsed as float, compare numerically
                                    input_float = float(input_val)
                                    prev_float = float(prev_result)
                                    if abs(input_float - prev_float) < 0.0001:
                                        formatted_inputs.append(str(step_results[prev_idx]))
                                        logger.info(f"Using result from step {prev_idx} as input {i}")
                                        is_previous_result = True
                                        break
                                except ValueError:
                                    # If not numeric, compare as strings
                                    if input_val == prev_result_str:
                                        formatted_inputs.append(str(step_results[prev_idx]))
                                        logger.info(f"Using result from step {prev_idx} as input {i}")
                                        is_previous_result = True
                                        break

                        if is_previous_result:
                            continue

                        # If not a previous result, use the mapped value
                        if input_val in number_mapping:
                            formatted_inputs.append(number_mapping[input_val])
                        else:
                            # Otherwise use the original value
                            formatted_inputs.append(input_val)

                    # Format the input string for the tool
                    formatted_input = ", ".join(formatted_inputs)
                    logger.info(f"Step {step_idx + 1}: Using {tool_name} with input {formatted_input}")

                    # Call the appropriate tool with the formatted input
                    try:
                        if tool_name == "sum":
                            result = math_toolbox.sum(formatted_input)
                        elif tool_name == "product":
                            result = math_toolbox.product(formatted_input)
                        elif tool_name == "divide":
                            result = math_toolbox.divide(formatted_input)
                        elif tool_name == "subtract":
                            result = math_toolbox.subtract(formatted_input)
                        elif tool_name == "power":
                            result = math_toolbox.power(formatted_input)
                        elif tool_name == "sqrt":
                            result = math_toolbox.sqrt(formatted_input)
                        elif tool_name == "modulo":
                            result = math_toolbox.modulo(formatted_input)
                        elif tool_name == "round_number":
                            result = math_toolbox.round_number(formatted_input)
                        else:
                            raise ValueError(f"Unknown tool: {tool_name}")

                        # Check if we got an error
                        if "Error occurred:" in result:
                            return f"Virtual tool execution failed at step {step_idx + 1}: {result}"

                        # Try to convert the result to a float if possible
                        try:
                            result_value = float(result)
                            step_results[step_idx] = result_value
                        except ValueError:
                            step_results[step_idx] = result

                        logger.info(f"Step {step_idx + 1} result: {step_results[step_idx]}")

                    except Exception as e:
                        error_msg = f"Step {step_idx + 1} failed: {str(e)}"
                        logger.error(error_msg)
                        return error_msg

                # Return the final result (from the last step)
                final_result = step_results.get(len(tool_sequence) - 1, "No result")
                return str(final_result)

            except Exception as e:
                logger.error(f"Virtual tool execution failed: {str(e)}")
                return f"Virtual tool execution failed: {str(e)}"

        # Register the virtual tool
        self.virtual_tools[problem_hash] = {
            "name": f"VirtualTool_{problem_type}_{problem_hash[:8]}",
            "description": f"Solves {problem_type} problems similar to: '{sequence_data['problem']}'",
            "function": virtual_tool_func,
            "primary_tool": primary_tool,  # Store the primary tool for reference
            "tool_sequence": tool_sequence  # Store the full sequence for reference
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

    # Add this method to the VirtualToolManager class

    def _is_tool_relevant_for_problem(self, problem: str, tool_sequence: List[Dict[str, Any]]) -> bool:
        """
        Determine if a tool sequence is relevant for a given problem.
        This helps prevent misapplication of complex sequences to simple problems.
        """
        # Check for simple multiplication problems
        is_simple_multiplication = re.search(r'multiply\s+(\d+)\s+by\s+(\d+)', problem.lower()) is not None

        if is_simple_multiplication:
            # For simple multiplication, don't use sequences with round_number
            for step in tool_sequence:
                if step["tool"] == "round_number":
                    return False

        # Add more specific filters as needed

        # By default, consider the tool relevant
        return True

    # Then modify the find_matching_virtual_tool method:

    def find_matching_virtual_tool(self, problem: str) -> Optional[Dict[str, Any]]:
        """Find a virtual tool that can solve a similar problem."""
        problem_hash = self.hash_problem(problem)

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

            # Skip tools with too many or too few operations
            tool_op_count = len(tool['tool_sequence'])
            if abs(tool_op_count - expected_op_count) > 1:  # Allow small variations
                logger.info(
                    f"Tool {tool['name']} has {tool_op_count} operations, but problem needs ~{expected_op_count}")
                continue

            # Check operation types
            if not self._has_compatible_operations(problem, tool['tool_sequence']):
                logger.info(f"Tool {tool['name']} has incompatible operation types for this problem")
                continue

            # Get structure similarity score
            original_problem_numbers = self._parse_expression(original_problem)
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
            return False
        if has_subtraction and tool_operations['subtract'] == 0:
            return False
        if has_multiplication and tool_operations['product'] == 0:
            return False
        if has_division and tool_operations['divide'] == 0:
            return False
        if has_power and tool_operations['power'] == 0:
            return False
        if has_square_root and tool_operations['sqrt'] == 0:
            return False

        # For subtraction specifically, check if the number of operations matches
        if has_subtraction:
            # Count subtraction operations in problem
            subtract_count = problem_lower.count('subtract') + problem_lower.count('minus') + problem_lower.count(
                '-')
            # If problem has clearly more subtractions than the tool can handle
            if subtract_count > tool_operations['subtract'] + 1:  # Allow some flexibility
                logger.info(
                    f"Problem needs ~{subtract_count} subtractions but tool only has {tool_operations['subtract']}")
                return False

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