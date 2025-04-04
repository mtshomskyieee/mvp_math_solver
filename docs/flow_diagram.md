flowchart TD
    A[User inputs problem] --> B{Check cache}
    B -->|Found in cache| C[Return cached solution]
    B -->|Not in cache| D{Check for virtual tool}
    
    D -->|Virtual tool found| E[Solve with virtual tool]
    D -->|No virtual tool| J[Solve with agents]
    
    E --> F{Verify solution}
    F -->|Solution verified| G[Return verified solution]
    F -->|Verification failed| H[Record tool failure]
    H --> I{Max failures reached?}
    I -->|Yes| W[Remove virtual tool]
    I -->|No| J
    
    J --> K[Solve with solver agent]
    J --> L[Solve with validation agent]
    J --> M[Solve with CAS agent]
    
    K & L & M --> N[Perform majority voting]
    N -->|No majority| O[Use solver's solution]
    N -->|Majority found| P[Use majority solution]
    
    O --> Q{Verify solution}
    P --> Q
    
    Q -->|Verified| R[Record successful sequence]
    Q -->|Not verified| S{Retry limit reached?}
    
    S -->|Yes| T[Return unverified solution]
    S -->|No| U[Retry with different approach]
    U --> K
    
    R --> V[Create virtual tool]
    V --> G
    
    G --> X[Cache result]
    X --> Y[Return final solution to user]
    T --> Y