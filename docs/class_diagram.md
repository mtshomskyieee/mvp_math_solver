classDiagram
    %% Core Classes
    class MathToolbox {
        -unreliable_tools: List[str]
        -max_unreliable: bool
        -tool_stats: Dict
        +set_all_tools_reliable()
        +unset_all_tools_reliable()
        +set_max_unreliable()
        +unset_max_unreliable()
        +get_tools_string()
        +sum(numbers_str)
        +product(numbers_str)
        +divide(numbers_str)
        +subtract(numbers_str)
        +power(numbers_str)
        +sqrt(number_str)
        +modulo(numbers_str)
        +round_number(input_str)
        +avg(numbers_str)
        +get_stats()
    }
    
    class VirtualToolManager {
        -virtual_tools: Dict
        -successful_sequences: Dict
        -tool_failure_counts: Dict
        -max_failures: int
        -vector_store: MathProblemVectorStore
        +hash_problem(problem)
        +record_successful_sequence(problem, sequence, result)
        +find_matching_virtual_tool(problem)
        +record_tool_failure(problem_hash)
        +save_virtual_tools_to_csv(filename)
        +import_virtual_tools_from_csv(filename)
        -_create_virtual_tool(problem_hash)
        -_categorize_problem(problem)
        -_map_numbers(new_numbers, original_numbers)
    }
    
    class MathProblemVectorStore {
        -embeddings: OpenAIEmbeddings
        -dimension: int
        -index: FAISS
        -problem_map: Dict
        -reverse_map: Dict
        +add_problem(problem, problem_hash, tool_sequence)
        +find_similar_problems(problem, k)
        +update_problem(problem_hash, problem, tool_sequence)
        +remove_problem(problem_hash)
        +save(filepath)
        +load(filepath)
    }
    
    %% Agent Classes
    class MathSolverAgent {
        -toolbox: MathToolbox
        -virtual_tool_manager: VirtualToolManager
        -execution_history: List
        -llm: ChatOpenAI
        -base_tools: List[Tool]
        -memory: ConversationBufferMemory
        -agent: Agent
        +streamlit_user_input(question)
        +solve_problem(problem, callback_handler)
    }
    
    class VerificationAgent {
        -toolbox: MathToolbox
        -llm: ChatOpenAI
        -tools: List[Tool]
        -memory: ConversationBufferMemory
        -agent: Agent
        +verify_result(problem, proposed_solution, callback_handler)
        -_extract_numeric_value(text)
        -_extract_nonnumeric_value(text)
        -_string_similarity(str1, str2)
        -_is_simple_arithmetic_problem(problem)
    }
    
    class CASAgent {
        -llm: ChatOpenAI
        +solve_problem(problem, callback_handler)
        +verify_result(problem, proposed_solution, callback_handler)
        -_extract_numeric_value(text)
        -_parse_equation(problem)
        -_solve_with_sympy(expression)
    }
    
    %% Callback and UI Classes
    class StreamlitCallbackHandler {
        -container: StreamlitContainer
        -text: str
        +on_llm_start(serialized, prompts)
        +on_llm_new_token(token)
        +on_tool_start(serialized, input_str)
        +on_tool_end(output)
        +on_agent_action(action)
    }
    
    %% Relationships
    MathSolverAgent --> MathToolbox : uses
    MathSolverAgent --> VirtualToolManager : uses
    VerificationAgent --> MathToolbox : uses
    VirtualToolManager --> MathProblemVectorStore : uses
    MathSolverAgent --> StreamlitCallbackHandler : uses
    VerificationAgent --> StreamlitCallbackHandler : uses
    CASAgent --> StreamlitCallbackHandler : uses