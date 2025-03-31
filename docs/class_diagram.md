classDiagram
    class StopException {
        +String message
        +new(message: &str) : StopException
        +fmt(&self, f: &mut fmt::Formatter) : fmt::Result
    }

    class MathToolbox {
        +Vec~String~ unreliable_tools
        +HashMap~String, HashMap~String, usize~~ tool_stats
        +bool errors_enabled
        +new() : MathToolbox
        +set_all_tools_reliable()
        +unset_all_tools_reliable()
        +toggle_errors()
        +get_errors_status() : &str
        +get_tools_string() : String
        +sum(numbers_str: &str) : String
        +product(numbers_str: &str) : String
        +divide(numbers_str: &str) : String
        +subtract(numbers_str: &str) : String
        +power(numbers_str: &str) : String
        +sqrt(number_str: &str) : String
        +modulo(numbers_str: &str) : String
        +round_number(input_str: &str) : String
        +get_stats() : &HashMap~String, HashMap~String, usize~~
    }

    class Tool {
        +String name
        +String description
    }

    class VirtualToolManager {
        +HashMap~String, VirtualTool~ virtual_tools
        +HashMap~String, SuccessfulSequence~ successful_sequences
        +HashMap~String, usize~ tool_failure_counts
        +usize max_failures
        +new(max_failures: usize) : VirtualToolManager
        +hash_problem(problem: &str) : String
        +record_successful_sequence(problem: &str, sequence: Vec~ToolStep~, result: &str)
        +_create_virtual_tool(problem_hash: &str)
        +find_matching_virtual_tool(problem: &str) : Option~&VirtualTool~
        +_is_tool_relevant_for_problem(problem: &str, tool_sequence: &[ToolStep]) : bool
        +_categorize_problem(problem: &str) : String
        +_parse_expression(expression: &str) : Vec~NumberWithContext~
        +_estimate_operation_count(problem: &str, numbers: &[NumberWithContext]) : usize
        +_has_compatible_operations(problem: &str, tool_sequence: &[ToolStep]) : bool
        +_calculate_structure_similarity(new_numbers: &[NumberWithContext], original_numbers: &[NumberWithContext]) : f64
        +_compare_operation_patterns(nums1: &[NumberWithContext], nums2: &[NumberWithContext]) : f64
        +record_tool_failure(problem_hash: &str) : bool
    }

    class VirtualTool {
        +String name
        +String description
        +Option~String~ primary_tool
        +Vec~ToolStep~ tool_sequence
    }

    class ToolStep {
        +String tool
        +String tool_input
    }

    class SuccessfulSequence {
        +String problem
        +Vec~ToolStep~ sequence
        +String result
        +chrono::DateTime~Utc~ created_at
    }

    class NumberWithContext {
        +String value
        +usize position
        +bool left_paren
        +bool right_paren
        +Option~char~ left_op
        +Option~char~ right_op
    }

    class Message {
        +String role
        +String content
    }

    class MathSolverAgent {
        +MathToolbox toolbox
        +VirtualToolManager virtual_tool_manager
        +Vec~ToolStep~ execution_history
        +new(toolbox: MathToolbox, virtual_tool_manager: VirtualToolManager) : MathSolverAgent
        +get_base_tools() : Vec~Tool~
        +solve_problem(problem: &str) : String
        +extract_tool_usage(response: &str, problem: &str)
    }

    class VerificationAgent {
        +MathToolbox toolbox
        +new(toolbox: MathToolbox) : VerificationAgent
        +verify_result(problem: &str, proposed_solution: &str) : (bool, String)
    }

    MathSolverAgent -- MathToolbox : uses
    MathSolverAgent -- VirtualToolManager : uses
    MathSolverAgent -- ToolStep : stores in execution_history
    MathSolverAgent -- Tool : creates and uses
    
    VerificationAgent -- MathToolbox : uses
    
    VirtualToolManager -- VirtualTool : manages
    VirtualToolManager -- SuccessfulSequence : stores
    VirtualToolManager -- ToolStep : uses in sequences
    VirtualToolManager -- NumberWithContext : uses for analysis
    
    VirtualTool -- ToolStep : contains sequence
    
    SuccessfulSequence -- ToolStep : stores sequence