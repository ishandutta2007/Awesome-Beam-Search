# Lookahead / Speculative Beam Decoding

Speculative Beam Decoding uses a smaller, faster draft model to propose beams and uses the target model to verify them in parallel.

## Workflow
1. Draft model unrolls multiple steps of beam search to produce a candidate tree.
2. Target model evaluates the tree in a single matrix-multiplication pass.
3. Accepted tokens are committed, and failed branches are discarded.

## Verification Pipeline Diagram
```mermaid
graph TD
    A[Draft Model: Fast Proposal] --> B[Draft Beam Tree]
    B --> C[Target Model: Parallel Verification]
    C --> D{Match?}
    D -- Yes --> E[Accept and Commit]
    D -- No --> F[Correct & Restart]
```

[Back to README](../README.md)
