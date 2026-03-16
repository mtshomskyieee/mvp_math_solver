# Math Solver Multi-Agent System

A Streamlit application that uses multiple AI agents to solve and verify math problems.

## Technologies
- Python + Streamlit
- Rust
- Docker
- Pair coded with AI (that's why there's a generate_all.sh)

# Project Structure
Core project files
- `config/`: Configuration settings
- `core/`: Core functionality (math toolbox, virtual tools)
- `agents/`: Agent implementations
- `utils/`: Utility functions
- `workflows/`: Workflow definitions
- `ui/`: Streamlit UI components
- `tests/`: Unit tests

Additional exploration:
- `api/`: fastapi interface to agents 
- `rust/`: ported rust implementation 

## Files
```
│
├── config/                    # Configuration files
│   ├── __init__.py
│   └── settings.py            # Contains app settings and API keys
│
├── core/                      # Core functionality
│   ├── __init__.py
│   ├── math_toolbox.py         # Math tools implementation
│   ├── virtual_tool_manager.py # Virtual tool creation and management
│   └── callbacks.py            # Callback handlers
│
├── agents/                    # Agent implementations
│   ├── __init__.py
│   ├── solver_agent.py        # Math solver agent
│   └── verification_agent.py  # Verification agent
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── logging_utils.py       # Logging configuration
│
├── workflows/                 # Workflow implementations
│   ├── __init__.py
│   └── math_workflow.py       # Main math solving workflow
│
├── ui/                        # UI components
│   ├── __init__.py
│   ├── main_app.py            # Main Streamlit app
│   ├── sidebar.py             # Sidebar components
│   └── problem_solver.py      # Problem solver UI
│
│
├── main.py                    # Application entry point
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Features

- Math problem solving with integrated verification
- Virtual tool creation for improved efficiency
- Real-time evaluation of solver performance
- Interactive user interface

## Architecture

This application uses a modular architecture with:

- Multiple specialized agents (solver, verification, cas)
- Virtual tool creation from successful problem-solving sequences
- Integrated validation and retry workflow

## Running using Docker
Run the following
- Set your OPENAI_API_KEY as an environment variable
- `docker compose build`
- `docker compose up`

## Running 

1. Setup a virtual environment `python3 -m venv venv` then `source venv/bin/activate`
   1. remember to run `source venv/bin/activate` every time you open a shell to this project to run
   2. ☝️ Yes, you must activate your environment to grab locally installed libs
2. Install dependencies: `pip install -r requirements.txt`
3. Set your OPENAI_API_KEY or edit main.py 's override_key with your key  
   1. ☝️☝Add your key
4. Run `streamlit run main.py`

## Running unit tests

To run unit tests, run the following:
- `pytest ./agents/test*py ./core/test*py`


# Diagrams

## Flow
![flow_diagram.png](docs%2Fflow_diagram.png)

## Sequence
![sequence.png](docs%2Fsequence.png)

## Class
![class_diagram.png](docs%2Fclass_diagram.png)

# Slides
[SpindleAI Application.pdf](docs%2FAgenticMath.pdf)

