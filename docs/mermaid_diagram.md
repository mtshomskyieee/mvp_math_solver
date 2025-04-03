graph TB
    %% Actors and Entry Points
    User([User]) -->|Enter math equation| UI[Streamlit UI]
    
    %% Main Workflow
    UI -->|Process equation| MathWorkflow[Math Workflow]
    
    %% Virtual Tool Check
    MathWorkflow -->|Check for similar problems| VTManager[Virtual Tool Manager]
    VTManager <-->|Query/Store problems| VStore[(Vector Store)]
    
    %% Solution Paths
    VTManager -->|Virtual tool found| VTExec[Execute Virtual Tool]
    VTManager -->|No virtual tool| SolverAgent[Math Solver Agent]
    
    %% Tool Execution
    VTExec -->|Execute operations| MathToolbox[Math Toolbox]
    SolverAgent -->|Make tool calls| MathToolbox
    
    %% Solution Generation
    MathToolbox -->|Return calculation results| VTExec
    MathToolbox -->|Return calculation results| SolverAgent
    
    %% Verification Process
    VTExec -->|Solution| VerificationAgent[Verification Agent]
    SolverAgent -->|Solution| VerificationAgent
    
    VerificationAgent -->|Verify using| MathToolbox
    VerificationAgent -->|Verification result| VerifyDecision{Verified?}
    
    %% Decision Branches
    VerifyDecision -->|Yes| Success[Record & Return Solution]
    VerifyDecision -->|No & retries left| Retry[Retry Solution]
    VerifyDecision -->|No & max retries| Failure[Return Best Attempt]
    
    Retry -->|Try again| SolverAgent
    
    %% Feedback Loop for Learning
    Success -->|Record successful sequence| VTManager
    
    %% Final Output
    Success -->|Return verified solution| UI
    Failure -->|Return best solution| UI
    
    UI -->|Display result| User
    
    %% Styling
    classDef component fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    classDef flow fill:#fff,stroke:#333,stroke-width:1px
    classDef decision fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
    classDef toolbox fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    
    class MathToolbox toolbox
    class UI,MathWorkflow,VTManager,VStore,SolverAgent,VerificationAgent component
    class User,VTExec,Success,Failure,Retry flow
    class VerifyDecision decision