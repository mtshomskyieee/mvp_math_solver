graph TD
    MainApp[Main App] --> ProblemSolver[Problem Solver UI]
    MainApp --> Sidebar[Sidebar UI]
    
    ProblemSolver --> MathWorkflow[Math Workflow]
    
    MathWorkflow --> MathSolverAgent
    MathWorkflow --> VerificationAgent
    MathWorkflow --> CASAgent
    MathWorkflow --> VirtualToolManager
    
    MathSolverAgent --> MathToolbox
    MathSolverAgent --> VirtualToolManager
    VerificationAgent --> MathToolbox
    
    VirtualToolManager --> MathProblemVectorStore
    
    MathSolverAgent --> StreamlitCallbackHandler
    VerificationAgent --> StreamlitCallbackHandler
    CASAgent --> StreamlitCallbackHandler
    
    MathToolbox --> RustMathExtensions[Rust Math Extensions]
    
    subgraph API
        OpenAIEmbeddings
    end
    
    MathProblemVectorStore --> OpenAIEmbeddings
    MathSolverAgent --> OpenAIEmbeddings
    VerificationAgent --> OpenAIEmbeddings
    CASAgent --> OpenAIEmbeddings