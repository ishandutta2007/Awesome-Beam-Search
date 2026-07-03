# Diverse Beam Search (DBS)

Diverse Beam Search partitions the beam budget into groups, penalizing overlap to generate highly diverse outputs.

## Group Formulation
Let $B$ be the beam width, divided into $G$ groups.
$$\text{Score}(y_{t, g}) = \text{Score}(y_{t, g-1}) + \lambda \cdot \text{DiversityPenalty}(y_{t, g}, Y_{t, <g})$$

## Advantages
- Higher token and semantic diversity.
- Explores alternative grammar structures.

## Group Decoding Diagram
```mermaid
flowchart TD
    subgraph Group 1
    A1[Explore Path A]
    end
    subgraph Group 2
    A2[Explore Path B - Penalty if overlaps with A]
    end
    Group 1 -.-> Group 2
```

[Back to README](../README.md)
