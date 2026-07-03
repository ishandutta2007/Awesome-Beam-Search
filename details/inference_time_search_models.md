# Native Inference-Time Search for Advanced Reasoning Models

Modern reasoning frameworks perform multi-path reasoning steps natively at inference time using self-correction and process reward feedback.

## Process
- Generates logical nodes.
- Traverses code execution or proof verification paths.
- Uses value networks to estimate the likelihood of reaching a correct answer.

## Search Tree Diagram
```mermaid
graph TD
    Root[Root Prompt] --> Step1[Step 1 Hypothesis A]
    Root --> Step2[Step 1 Hypothesis B]
    Step1 --> Step1_1[Step 2 Successor]
    Step2 -->|Value Network Prunes| Prune[Pruned/Failed Branch]
```

[Back to README](../README.md)
