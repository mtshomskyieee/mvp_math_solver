import pytest
from agents.verification_agent import VerificationAgent
from core.math_toolbox import MathToolbox

@pytest.fixture
def math_toolbox():
    return MathToolbox()

@pytest.fixture
def verification_agent(math_toolbox):
    return VerificationAgent(math_toolbox)

def test_verification_agent_initialization(verification_agent):
    assert verification_agent is not None
    assert len(verification_agent.tools) == 9  # Verify all tools are present

def test_verify_numeric_addition_result(verification_agent):
    problem = "What is 5 + 3?"
    solution = "8"
    is_verified, verification_text = verification_agent.verify_result(problem, solution)
    assert is_verified is True
    assert "VERIFIED" in verification_text

def test_verify_incorrect_solution(verification_agent):
    problem = "What is 5 + 3?"
    solution = "9"
    is_verified, verification_text = verification_agent.verify_result(problem, solution)
    assert is_verified is False
    assert "INCORRECT" in verification_text

def test_verify_complex_number_result(verification_agent):
    problem = "What is the square root of -1?"
    solution = "i"
    is_verified, verification_text = verification_agent.verify_result(problem, solution)
    assert is_verified is True
    assert "complex number" in verification_text.lower()