sequenceDiagram
    participant User
    participant MainApp
    participant MathWorkflow
    participant MathSolverAgent
    participant VirtualToolManager
    participant VerificationAgent
    participant CASAgent
    participant MathToolbox
    
    User->>MainApp: Enter math problem
    MainApp->>MathWorkflow: math_workflow(problem)
    
    MathWorkflow->>VirtualToolManager: find_matching_virtual_tool(problem)
    
    alt Virtual Tool Found
        VirtualToolManager-->>MathWorkflow: Return matching virtual tool
        MathWorkflow->>MathToolbox: Execute virtual tool function
        MathToolbox-->>MathWorkflow: Return solution
        MathWorkflow->>VerificationAgent: verify_result(problem, solution)
        
        alt Verification Success
            VerificationAgent-->>MathWorkflow: Return verified=true
        else Verification Failure
            VerificationAgent-->>MathWorkflow: Return verified=false
            VirtualToolManager->>VirtualToolManager: record_tool_failure(problem_hash)
            MathWorkflow->>MathSolverAgent: solve_problem(problem)
        end
    else No Virtual Tool
        MathWorkflow->>MathSolverAgent: solve_problem(problem)
        MathSolverAgent->>MathToolbox: Use math tools (sum, product, etc.)
        MathToolbox-->>MathSolverAgent: Return tool results
        MathSolverAgent-->>MathWorkflow: Return solution
        
        MathWorkflow->>VerificationAgent: verify_result(problem, solution)
        MathWorkflow->>CASAgent: solve_problem(problem)
        CASAgent-->>MathWorkflow: Return CAS solution
        
        MathWorkflow->>MathWorkflow: Perform majority voting on solutions
        
        MathWorkflow->>VirtualToolManager: record_successful_sequence(problem, sequence, result)
        VirtualToolManager->>VirtualToolManager: _create_virtual_tool(problem_hash)
        VirtualToolManager->>MathProblemVectorStore: add_problem(problem, problem_hash, sequence)
    end
    
    MathWorkflow-->>MainApp: Return final solution and verification status
    MainApp-->>User: Display solution