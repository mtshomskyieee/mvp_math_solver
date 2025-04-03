flowchart TD
    User([User]) -->|Enter math equation| UI[Streamlit UI]
    UI -->|Submit problem| Workflow[Math Workflow]
    
    Workflow -->|Check| Cache{Cache lookup}
    Cache -->|Hit| CachedSolution[Return cached solution]
    Cache -->|Miss| VirtualTool{Virtual tool match?}
    
    VirtualTool -->|Yes| ExecuteVT[Execute virtual tool]
    VirtualTool -->|No| SolverAgent[Math Solver Agent]
    
    SolverAgent -->|Use| Toolbox[Math Toolbox]
    SolverAgent -->|Generate| Solution[Proposed solution]
    
    ExecuteVT -->|Produce| VTSolution[Virtual tool solution]
    
    Solution --> Verification[Verification Agent]
    VTSolution --> Verification
    
    Verification -->|Validate| ValidCheck{Is solution valid?}
    ValidCheck -->|Yes| RecordPattern[Record solution pattern]
    ValidCheck -->|No, retries left| RetryAttempt[Retry solution]
    ValidCheck -->|No, no retries| FinalAttempt[Return best attempt]
    
    RetryAttempt --> SolverAgent
    
    RecordPattern -->|Create/update| CreateVT[Virtual Tool Manager]
    RecordPattern -->|Store in| UpdateCache[Cache]
    
    CachedSolution --> DisplayResult[Display result to user]
    RecordPattern --> DisplayResult
    FinalAttempt --> DisplayResult
    
    DisplayResult --> User