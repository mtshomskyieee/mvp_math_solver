classDiagram
    class MathToolbox {
        -unreliable_tools: List
        -max_unreliable: bool
        -tool_stats: Dict
        +set_all_tools_reliable()
        +unset_all_tools_reliable()
        +set_max_unreliable()
        +unset_max_unreliable()
        +get_tools_string() String
        +sum(numbers_str) String
        +product(numbers_str) String
        +divide(numbers_str) String
        +subtract(numbers_str) String
        +power(numbers_str) String
        +sqrt(number_str) String
        +modulo(numbers_str) String
        +round_number(input_str) String
        +avg(numbers_str) String
        +get_stats() Dict
    }

    class VirtualToolManager {
        -virtual_tools: Dict
        -successful_sequences: Dict
        -tool_failure_counts: Dict
        -max_failures: int
        -vector_store: MathProblemVectorStore
        +hash_problem(problem) String
        +record_successful_sequence(problem, sequence, result)
        +record_tool_failure(problem_hash) Boolean
        +find_matching_virtual_tool(problem) Dict
        +save_virtual_tools_to_csv(filename)
        +import_virtual_tools_from_csv(filename)
        +migrate_existing_tools_to_vector_store()
    }

    class MathProblemVectorStore {
        -embeddings: OpenAIEmbeddings
        -dimension: int
        -index: FaissIndex
        -problem_map: Dict
        -reverse_map: Dict
        +add_problem(problem, problem_hash, tool_sequence)
        +find_similar_problems(problem, k) List
        +update_problem(problem_hash, problem, tool_sequence)
        +remove_problem(problem_hash) Boolean
        +save(filepath)
        +load(filepath) Boolean
    }

    class StreamlitCallbackHandler {
        -container: Object
        -text: String
        +on_llm_start(serialized, prompts)
        +on_llm_new_token(token)
        +on_tool_start(serialized, input_str)
        +on_tool_end(output)
        +on_agent_action(action)
    }

    class MathSolverAgent {
        -toolbox: MathToolbox
        -virtual_tool_manager: VirtualToolManager
        -execution_history: List
        -llm: ChatOpenAI
        -base_tools: List[Tool]
        -memory: ConversationBufferMemory
        -agent: Agent
        +streamlit_user_input(question) String
        +solve_problem(problem, callback_handler) String
    }

    class VerificationAgent {
        -toolbox: MathToolbox
        -llm: ChatOpenAI
        -tools: List[Tool]
        -memory: ConversationBufferMemory
        -agent: Agent
        +verify_result(problem, proposed_solution, callback_handler) Tuple
    }

    class CASAgent {
        -llm: ChatOpenAI
        +_extract_numeric_value(text) Number
        +_parse_equation(problem) String
        +_solve_with_sympy(expression) String
        +solve_problem(problem, callback_handler) String
        +verify_result(problem, proposed_solution, callback_handler) Tuple
    }

    MathSolverAgent --> MathToolbox
    MathSolverAgent --> VirtualToolManager
    VerificationAgent --> MathToolbox
    VirtualToolManager --> MathProblemVectorStore