sequenceDiagram
    actor User
    participant Workflow
    participant VTManager as VirtualToolManager
    participant Solver as SolverAgent
    participant Verifier as VerificationAgent
    participant Tools as MathToolbox

    User->>Workflow: Enter math equation
    Workflow->>Workflow: Check cache
    
    alt Found in cache
        Workflow-->>User: Return cached solution
    else Not in cache
        Workflow->>VTManager: Find matching virtual tool
        
        alt Virtual tool exists
            VTManager-->>Workflow: Return virtual tool
            Workflow->>Tools: Execute virtual tool
            Tools-->>Workflow: Result
            Workflow->>Verifier: Verify result
            
            alt Verification successful
                Verifier-->>Workflow: Result verified
                Workflow-->>User: Display verified solution
            else Verification failed
                Verifier-->>Workflow: Verification failed
                Workflow->>VTManager: Record tool failure
                Workflow->>Solver: Fallback to standard solving
            end
        else No virtual tool
            VTManager-->>Workflow: No tool found
            Workflow->>Solver: Solve problem
        end
        
        Solver->>Tools: Use math tools
        Tools-->>Solver: Tool results
        Solver-->>Workflow: Solution
        
        Workflow->>Verifier: Verify solution
        
        alt Verification successful
            Verifier-->>Workflow: Solution verified
            Workflow->>VTManager: Record successful sequence
            Workflow->>Workflow: Cache result
            Workflow-->>User: Display verified solution
        else Verification failed
            Verifier-->>Workflow: Verification failed
            
            loop Until MAX_RETRIES or success
                Workflow->>Solver: Retry solving
                Solver->>Tools: Use math tools
                Tools-->>Solver: Tool results
                Solver-->>Workflow: New solution
                
                Workflow->>Verifier: Verify new solution
                
                alt Verification successful
                    Verifier-->>Workflow: Solution verified
                    Workflow->>VTManager: Record successful sequence
                    Workflow->>Workflow: Cache result
                    Workflow-->>User: Display verified solution
                    note over Workflow: Exit retry loop
                end
            end
            
            Workflow-->>User: Display best attempt
        end
    end