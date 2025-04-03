import pytest
from unittest.mock import MagicMock, patch
from agents.cas_agent import CASAgent


@pytest.fixture
def cas_agent():
    return CASAgent()


def test_cas_agent_initialization(cas_agent):
    assert cas_agent is not None
    assert hasattr(cas_agent, 'llm')


def test_extract_numeric_value(cas_agent):
    # Test extracting regular numbers
    assert cas_agent._extract_numeric_value("The result is 42") == 42.0
    assert cas_agent._extract_numeric_value("-3.14") == -3.14

    # Test extracting complex numbers
    assert cas_agent._extract_numeric_value("i") == complex(0, 1)
    assert cas_agent._extract_numeric_value("-2i") == complex(0, -2)
    assert cas_agent._extract_numeric_value("3i") == complex(0, 3)

    # Test no numeric value
    assert cas_agent._extract_numeric_value("No numbers here") is None


def test_parse_equation(cas_agent):
    # Instead of patching the 'invoke' method directly, create a mock response
    mock_response = MagicMock()
    mock_response.content = "5 + 3"

    # Patch the entire method with a mock that returns our predefined response
    with patch.object(cas_agent.llm.__class__, 'invoke', return_value=mock_response):
        result = cas_agent._parse_equation("What is 5 plus 3?")
        assert result == "5 + 3"

def test_solve_with_sympy_basic_operations(cas_agent):
    # Test addition
    assert cas_agent._solve_with_sympy("5 + 3") == "8"

    # Test subtraction
    assert cas_agent._solve_with_sympy("10 - 4") == "6"

    # Test multiplication
    assert cas_agent._solve_with_sympy("6 * 7") == "42"

    # Test division
    assert int(float(cas_agent._solve_with_sympy("20 / 5"))) == int(float("4.0"))


def test_solve_with_sympy_complex_operations(cas_agent):
    # Test power
    assert cas_agent._solve_with_sympy("2**3") == "8"

    # Test square root of positive number
    result = cas_agent._solve_with_sympy("sqrt(16)")
    assert result == "4" or result == "4.0"

    # Test square root of negative number
    complex_result = cas_agent._solve_with_sympy("sqrt(-4)")
    assert "i" in complex_result or "2j" in complex_result

    # Test modulo
    assert cas_agent._solve_with_sympy("10 % 3") == "1"

    # Test rounding
    assert cas_agent._solve_with_sympy("round(3.7)") == "4"


def test_solve_with_sympy_combined_operations(cas_agent):
    # Test complex expression
    assert cas_agent._solve_with_sympy("(2 + 3) * 4") == "20"

    # Test another complex expression
    result = cas_agent._solve_with_sympy("(10 / 2) + (3 * 4)")
    assert float(result) == 17.0


def test_solve_problem(cas_agent):
    with patch.object(cas_agent, '_parse_equation') as mock_parse:
        with patch.object(cas_agent, '_solve_with_sympy') as mock_solve:
            # Setup mocks
            mock_parse.return_value = "5 + 3"
            mock_solve.return_value = "8"

            # Test without callback
            result = cas_agent.solve_problem("What is 5 plus 3?")
            assert result == "8"

            # Test with callback
            callback = MagicMock()
            callback.container = MagicMock()
            result = cas_agent.solve_problem("What is 5 plus 3?", callback)
            assert result == "8"
            assert callback.container.markdown.call_count > 0
            assert callback.container.write.call_count > 0


def test_verify_result(cas_agent):
    with patch.object(cas_agent, 'solve_problem') as mock_solve:
        # Setup mock
        mock_solve.return_value = "8"

        # Test correct answer
        is_verified, message = cas_agent.verify_result("What is 5 plus 3?", "8")
        assert is_verified is True
        assert "VERIFIED" in message

        # Test incorrect answer
        is_verified, message = cas_agent.verify_result("What is 5 plus 3?", "9")
        assert is_verified is False
        assert "INCORRECT" in message

        # Test complex number verification
        mock_solve.return_value = "2i"
        is_verified, message = cas_agent.verify_result("What is the square root of -4?", "2i")
        assert is_verified is True

        # Test with callback - Reset the mock to return "8" again
        mock_solve.return_value = "8"

        # Mock the extract_numeric_value method to ensure consistent comparison
        with patch.object(cas_agent, '_extract_numeric_value', side_effect=lambda x: float(8) if '8' in x else None):
            callback = MagicMock()
            callback.container = MagicMock()
            is_verified, message = cas_agent.verify_result("What is 5 plus 3?", "8", callback)
            assert is_verified is True

def test_error_handling(cas_agent):
    with patch.object(cas_agent, '_parse_equation') as mock_parse:
        # Test error in parsing
        mock_parse.side_effect = Exception("Test error")
        result = cas_agent.solve_problem("Bad problem")
        assert "Failed to solve" in result

        # Test error in verification
        with patch.object(cas_agent, 'solve_problem') as mock_solve:
            mock_solve.side_effect = Exception("Test error")
            is_verified, message = cas_agent.verify_result("Bad problem", "42")
            assert is_verified is False
            assert "error" in message.lower()