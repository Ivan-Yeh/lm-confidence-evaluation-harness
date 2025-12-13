# LM Confidence Evaluation Harness

## Evaluation Framework Logic

```mermaid
flowchart LR
    A[Config] -->|obj| B[Dataset Manager]
    B -->|PromptCollection| C[Prompt Formatter]
    C -->|PromptCollection| D[Model Manager<br/>Runs Queries]
    D -->|list[ModelOutputs]| E[Output Post-processing<br/>Function]
    E -->|list[ModelOutputs]| F[Confidence Extraction<br/>Function]
    F -->|OrganisedOutputs| G[Grading]
    G -->|OrganisedOutputs| H[Metrics]
```

