# Logit Bias Modifier Layers

Logit Bias Modifier Layers dynamically intercept language model outputs by adding scalar penalties or rewards to logits before sampling.

## Equation
$$\tilde{z}_i = z_i + \text{bias}_i$$

## Interaction Diagram
```mermaid
graph LR
    A[Model Raw Logits] --> B[Apply Logit Bias Mask]
    B --> C[Softmax Function]
    C --> D[Beam Search Selection]
```

[Back to README](../README.md)
