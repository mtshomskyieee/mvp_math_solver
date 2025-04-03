import pytest
from unittest.mock import MagicMock, patch
from agents.solver_agent import MathSolverAgent
from core.math_toolbox import MathToolbox
from core.virtual_tool_manager import VirtualToolManager


@pytest.fixture
def math_toolbox():
    return MathToolbox()


@pytest.fixture
def virtual_tool_manager():
    return VirtualToolManager()


@pytest.fixture
def solver_agent(math_toolbox, virtual_tool_manager):
    return MathSolverAgent(math_toolbox, virtual_tool_manager)


def test_solver_agent_initialization(solver_agent):
    assert solver_agent is not None
    assert len(solver_agent.base_tools) == 10  # Verify all tools are present


def test_solve_problem_with_simple_addition(solver_agent):
    problem = "What is 5 + 3?"
    result = solver_agent.solve_problem(problem)
    assert result is not None
    assert "8" in result


def test_solve_problem_with_multiplication(solver_agent):
    problem = "Multiply 4 by 6"
    result = solver_agent.solve_problem(problem)
    assert result is not None
    assert "24" in result


def test_virtual_tool_usage(solver_agent):
    # Mock a virtual tool
    solver_agent.virtual_tool_manager.find_matching_virtual_tool = MagicMock(return_value={
        "name": "TestVirtualTool",
        "function": lambda input_str, math_toolbox: "10"
    })

    problem = "A test problem"
    result = solver_agent.solve_problem(problem)
    assert "10" in result