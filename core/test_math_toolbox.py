import pytest
import math
from core.math_toolbox import MathToolbox

@pytest.fixture
def math_toolbox():
    toolbox = MathToolbox()
    toolbox.set_all_tools_reliable()  # Ensure consistent results
    return toolbox

def test_sum_function(math_toolbox):
    result = math_toolbox.sum("1, 2, 3")
    assert int(float(result)) == 6

def test_product_function(math_toolbox):
    result = math_toolbox.product("2, 3, 4")
    assert int(float(result)) == 24

def test_divide_function(math_toolbox):
    result = math_toolbox.divide("10, 2")
    assert int(float(result)) == 5

def test_divide_by_zero(math_toolbox):
    result = math_toolbox.divide("10, 0")
    assert "Error" in result

def test_sqrt_function(math_toolbox):
    result = math_toolbox.sqrt("16")
    assert int(float(result)) == 4

def test_sqrt_negative_number(math_toolbox):
    result = math_toolbox.sqrt("-1")
    assert "i" in result

def test_modulo_function(math_toolbox):
    result = math_toolbox.modulo("10, 3")
    assert int(float(result)) == 1

def test_tool_stats(math_toolbox):
    math_toolbox.sum("1, 2, 3")
    stats = math_toolbox.get_stats()
    assert stats['sum']['calls'] == 1

def test_unreliable_tools(math_toolbox):
    math_toolbox.unset_all_tools_reliable()
    # Multiple calls might result in different outcomes
    results = [math_toolbox.sum("1, 2, 3") for _ in range(50)]
    # Assert we got results
    assert len(set(results)) > 1
    # Assert we hit one of our predefined error outputs
    assert (("Error occurred: Invalid input format" in results) or
            ("Error occurred: Could not parse numbers" in results))