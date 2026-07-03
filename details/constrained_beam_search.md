# Constrained / Lexical Beam Search

Constrained Beam Search enforces the inclusion of specific target words or phrases (lexical constraints) in the output.

## Strategy
- Maintain grid states tracking how many constraints have been satisfied.
- Prune paths that cannot satisfy remaining constraints in remaining tokens.

## Grid Tracking Diagram
```mermaid
graph TD
    A[No Constraints Met] -->|Include Keyword A| B[Constraint 1 Met]
    B -->|Include Keyword B| C[Constraint 1 & 2 Met]
    A -->|Prune if deadline reached| D[Failed Path]
```

[Back to README](../README.md)
