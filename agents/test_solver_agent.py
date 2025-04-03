import pytest
from unittest.mock import MagicMock, patch
from agents.solver_agent import MathSolverAgent
from core.math_toolbox import MathToolbox
from core.virtual_tool_manager import VirtualToolManager


@pytest.fixture
def math_toolbox():
    return MathToolbox()

@pytest.fixture
def reliable_math_toolbox():
    mt = MathToolbox()
    mt.set_all_tools_reliable()
    return mt

@pytest.fixture
def unreliable_math_toolbox():
    mt = MathToolbox()
    mt.unset_all_tools_reliable()
    return mt



@pytest.fixture
def virtual_tool_manager():
    return VirtualToolManager()


@pytest.fixture
def solver_agent(math_toolbox, virtual_tool_manager):
    return MathSolverAgent(math_toolbox, virtual_tool_manager)


def test_solver_agent_initialization(solver_agent):
    assert solver_agent is not None
    assert len(solver_agent.base_tools) == 10  # Verify all tools are present



def test_solve_problem_with_reliable_addition_repeated(reliable_math_toolbox, virtual_tool_manager):
    solver_agent = MathSolverAgent(reliable_math_toolbox, virtual_tool_manager)
    result_list = []
    for i in range(1,200):
        problem = "What is 40 + 2?"
        result = solver_agent.solve_problem(problem)
        assert result is not None
        result_list.append(result)
    for r in result_list:
        assert "42" in r

def test_solve_problem_with_unreliable_addition_repeated(unreliable_math_toolbox, virtual_tool_manager):
    solver_agent = MathSolverAgent(unreliable_math_toolbox, virtual_tool_manager)
    expected_bad_data_created = False
    for i in range(1,1000):
        problem = "What is 40 + 2?"
        result = solver_agent.solve_problem(problem)
        assert result is not None
        if "42" not in result:
            expected_bad_data_created = True
            break

    assert expected_bad_data_created == True



def test_solve_problem_with_reliable_multiplication_repeated(reliable_math_toolbox, virtual_tool_manager):
    solver_agent = MathSolverAgent(reliable_math_toolbox, virtual_tool_manager)
    result_list = []
    for i in range(1, 200):
        problem = "Multiply 4 by 6"
        result = solver_agent.solve_problem(problem)
        assert result is not None
        result_list.append(result)
    for r in result_list:
        assert "24" in r



def test_solve_problem_with_unreliable_multiplication_repeated(unreliable_math_toolbox, virtual_tool_manager):
    solver_agent = MathSolverAgent(unreliable_math_toolbox, virtual_tool_manager)
    expected_bad_data_created = False
    for i in range(1, 1000):
        problem = "Multiply 4 by 6"
        result = solver_agent.solve_problem(problem)
        assert result is not None
        if "24" not in result:
            expected_bad_data_created = True
            break
    assert expected_bad_data_created == True


def test_virtual_tool_usage(solver_agent):
    # Mock a virtual tool
    solver_agent.virtual_tool_manager.find_matching_virtual_tool = MagicMock(return_value={
        "name": "TestVirtualTool",
        "function": lambda input_str, math_toolbox: "10"
    })

    problem = "A test problem"
    result = solver_agent.solve_problem(problem)
    assert "10" in result