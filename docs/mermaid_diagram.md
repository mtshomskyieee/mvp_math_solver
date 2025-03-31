graph TB
    subgraph config
        settings[settings.py]
    end

    subgraph core
        mathToolbox[MathToolbox]
        vtm[VirtualToolManager]
        callbacks[Callbacks]
    end

    subgraph agents
        solverAgent[SolverAgent]
        verificationAgent[VerificationAgent]
    end

    subgraph workflows
        mathWorkflow[MathWorkflow]
    end

    subgraph ui
        mainApp[MainApp]
        sidebar[Sidebar]
        problemSolver[ProblemSolver]
    end

    subgraph utils
        loggingUtils[LoggingUtils]
        exceptions[Exceptions]
    end

    %% Main entry point
    main[main.py] --> mainApp

    %% UI connections
    mainApp --> sidebar
    mainApp --> problemSolver
    sidebar --> mathToolbox
    sidebar --> vtm
    problemSolver --> mathWorkflow

    %% Workflow connections
    mathWorkflow --> solverAgent
    mathWorkflow --> verificationAgent
    mathWorkflow --> vtm

    %% Agent connections
    solverAgent --> mathToolbox
    solverAgent --> vtm
    solverAgent --> callbacks
    verificationAgent --> mathToolbox

    %% Configuration and utilities
    settings --> solverAgent
    settings --> verificationAgent
    exceptions --> mathWorkflow
    exceptions --> solverAgent
    loggingUtils --> mathToolbox
    loggingUtils --> vtm
    loggingUtils --> solverAgent
    loggingUtils --> verificationAgent
    loggingUtils --> mathWorkflow

    %% Data flow
    problemSolver -- Problem Input --> mathWorkflow
    mathWorkflow -- Solution Result --> problemSolver

    classDef configClass fill:#f9f,stroke:#333,stroke-width:2px;
    classDef coreClass fill:#bbf,stroke:#333,stroke-width:2px;
    classDef agentClass fill:#bfb,stroke:#333,stroke-width:2px;
    classDef uiClass fill:#fbb,stroke:#333,stroke-width:2px;
    classDef utilClass fill:#ffffb2,stroke:#333,stroke-width:2px;
    classDef workflowClass fill:#b2ffb2,stroke:#333,stroke-width:2px;

    class settings configClass;
    class mathToolbox,vtm,callbacks coreClass;
    class solverAgent,verificationAgent agentClass;
    class mainApp,sidebar,problemSolver uiClass;
    class loggingUtils,exceptions utilClass;
    class mathWorkflow workflowClass;