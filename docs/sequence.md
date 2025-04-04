sequenceDiagram
    participant User
    participant App as Main App
    participant Workflow as Math Workflow
    participant SolverAgent as Math Solver Agent
    participant VerificationAgent
    participant CASAgent
    participant VTM as Virtual Tool Manager
    participant MT as Math Toolbox
    participant VS as Vector Store

    User->>App: Enter math problem
    App->>Workflow: Solve problem
    
    Workflow->>VTM: Check for virtual tool
    VTM->>VS: Find similar problems
    VS-->>VTM: Return matching tools
    
    alt Virtual tool found
        VTM-->>Workflow: Return virtual tool
        Workflow->>MT: Execute virtual tool function
        MT-->>Workflow: Return result
        Workflow->>VerificationAgent: Verify result
        VerificationAgent-->>Workflow: Verification result
    else No virtual tool found
        Workflow->>SolverAgent: Solve problem
        SolverAgent->>MT: Use math tools
        MT-->>SolverAgent: Tool results
        SolverAgent-->>Workflow: Solver solution
        
        Workflow->>VerificationAgent: Get solution
        VerificationAgent-->>Workflow: Validation solution
        
        Workflow->>CASAgent: Solve problem
        CASAgent-->>Workflow: CAS solution
        
        Workflow->>Workflow: Perform majority voting
        
        Workflow->>VerificationAgent: Verify final solution
        VerificationAgent-->>Workflow: Verification result
        
        alt Solution verified
            Workflow->>VTM: Record successful sequence
            VTM->>VS: Add problem to vector store
        end
    end
    
    Workflow-->>App: Return final solution
    App-->>User: Display solution