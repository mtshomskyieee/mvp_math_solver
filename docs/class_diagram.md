classDiagram
    class MathToolbox {
        -unreliable_tools: List[str]
        -tool_stats: Dict
        +set_all_tools_reliable()
        +unset_all_tools_reliable()
        +get_tools_string() str
        +sum(numbers_str) str
        +product(numbers_str) str
        +divide(numbers_str) str
        +subtract(numbers_str) str
        +power(numbers_str) str
        +sqrt(number_str) str
        +modulo(numbers_str) str
        +round_number(input_str) str
        +avg(numbers_str) str
        +get_stats() Dict
    }

    class MathProblemVectorStore {
        -embeddings: OpenAIEmbeddings
        -dimension: int
        -index: faiss.IndexFlatL2
        -problem_map: Dict
        -reverse_map: Dict
        +add_problem(problem, problem_hash, tool_sequence)
        +find_similar_problems(problem, k) List
        +update_problem(problem_hash, problem, tool_sequence)
        +remove_problem(problem_hash) bool
        +save(filepath)
        +load(filepath) bool
    }

    class VirtualToolManager {
        -virtual_tools: Dict
        -successful_sequences: Dict
        -tool_failure_counts: Dict
        -max_failures: int
        -vector_store: MathProblemVectorStore
        +hash_problem(problem) str
        +record_successful_sequence(problem, sequence, result)
        +find_matching_virtual_tool(problem) Dict
        +record_tool_failure(problem_hash) bool
        +serialize_virtual_tool(problem_hash) str
        +save_virtual_tools_to_csv(filename)
        +import_virtual_tools_from_csv(filename) int
        +migrate_existing_tools_to_vector_store() int
        -_create_virtual_tool(problem_hash)
        -_optimize_tool_sequence(tool_sequence) List
    }

    class MathSolverAgent {
        -toolbox: MathToolbox
        -virtual_tool_manager: VirtualToolManager
        -execution_history: List
        -llm: ChatOpenAI
        -base_tools: List[Tool]
        -memory: ConversationBufferMemory
        -agent: Agent
        +streamlit_user_input(question) str
        +solve_problem(problem, callback_handler) str
    }

    class VerificationAgent {
        -toolbox: MathToolbox
        -llm: ChatOpenAI
        -tools: List[Tool]
        -memory: ConversationBufferMemory
        -agent: Agent
        +verify_result(problem, proposed_solution, callback_handler) Tuple[bool, str]
        -_extract_numeric_value(text) float
        -_extract_nonnumeric_value(text) str
        -_string_similarity(str1, str2) float
        -_is_simple_arithmetic_problem(problem) bool
    }

    class BaseCallbackHandler {
        +on_llm_start()
        +on_llm_new_token()
    }

    class StreamlitCallbackHandler {
        -container: Container
        -text: str
        +on_llm_start(serialized, prompts)
        +on_llm_new_token(token)
        +on_tool_start(serialized, input_str)
        +on_tool_end(output)
        +on_agent_action(action)
    }

    class StopException {
    }

    class math_workflow {
        +math_workflow(problem, solver_agent, verification_agent, vtm, callback_handler) Dict
    }

    class UI_Components {
        +problem_input_section()
        +solve_problem_section()
        +display_solution_results()
        +render_sidebar()
        +app()
    }

    MathSolverAgent o-- MathToolbox : uses
    MathSolverAgent o-- VirtualToolManager : uses
    VerificationAgent o-- MathToolbox : uses
    VirtualToolManager o-- MathProblemVectorStore : contains
    StreamlitCallbackHandler --|> BaseCallbackHandler : inherits
    StopException --|> Exception : inherits
    math_workflow ..> MathSolverAgent : uses
    math_workflow ..> VerificationAgent : uses
    math_workflow ..> VirtualToolManager : uses
    UI_Components ..> math_workflow : calls