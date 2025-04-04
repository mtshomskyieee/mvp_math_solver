flowchart TD
    Start([Start]) --> InputProblem[Input Math Problem]
    InputProblem --> CheckVirtualTool{Check for\nvirtual tool?}
    
    CheckVirtualTool -->|Found| UseVirtualTool[Use Virtual Tool]
    CheckVirtualTool -->|Not Found| SolveWithAgents[Solve with Agents]
    
    UseVirtualTool --> VerifySolution{Verify\nSolution}
    
    SolveWithAgents --> SolverAgentProcess[Math Solver Agent Process]
    SolverAgentProcess --> ValidationAgentProcess[Validation Agent Process] 
    ValidationAgentProcess --> CASAgentProcess[CAS Agent Process]
    CASAgentProcess --> MajorityVoting[Majority Voting]
    
    MajorityVoting --> VerifySolution
    
    VerifySolution -->|Verified| RecordSequence[Record Successful Sequence]
    VerifySolution -->|Not Verified\n< Max Retries| RetryAttempt[Retry with Different Approach]
    RetryAttempt --> SolveWithAgents
    
    VerifySolution -->|Not Verified\n>= Max Retries| ContinueAnyway[Use Best Solution]
    
    RecordSequence --> CreateVirtualTool[Create Virtual Tool]
    CreateVirtualTool --> UpdateVectorStore[Update Vector Store]
    
    ContinueAnyway --> ReturnResult[Return Result]
    UpdateVectorStore --> ReturnResult
    
    ReturnResult --> DisplaySolution[Display Solution]
    DisplaySolution --> End([End])