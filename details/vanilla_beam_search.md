# Vanilla Beam Search

Vanilla Beam Search is a classical heuristic graph search algorithm that maintains a set of $B$ best candidate paths at any point in sequence decoding.

## Score Formulation
$$\text{Score}(Y) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, X)$$

## Mechanics
- Tracks exact joint probabilities.
- Deterministic and local-context sensitive.

## Process Diagram
```mermaid
graph LR
    Start --> B1[Path 1]
    Start --> B2[Path 2]
    B1 --> B1_1[Path 1.1]
    B1 --> B1_2[Path 1.2]
    B2 --> B2_1[Path 2.1]
    B2 --> B2_2[Path 2.2]
    B1_1 & B1_2 & B2_1 & B2_2 --> Score[Compute & Sort Scores]
    Score --> Prune[Retain Top B]
```

[Back to README](../README.md)
