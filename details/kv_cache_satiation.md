# The Key-Value Cache VRAM Satiation Wall

Long-context generation combined with large beam widths requires storing massive KV cache tensors, saturating GPU high bandwidth memory (HBM).

## Solutions
- **PagedAttention:** Virtualizing the KV cache to eliminate external fragmentation.
- **Grouped-Query Attention (GQA):** Reducing the query head size compared to key-value heads.

## Memory Mitigation Diagram
```mermaid
graph TD
    A[Continuous KV Cache] -->|PagedAttention Virtualization| B[Fragmented Memory Blocks]
    B --> C[HBM Saving & Higher Throughput]
```

[Back to README](../README.md)
