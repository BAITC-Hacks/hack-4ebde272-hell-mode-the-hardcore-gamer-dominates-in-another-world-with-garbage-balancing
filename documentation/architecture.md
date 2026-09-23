# Money Graph architecture

```mermaid
flowchart LR
    P[Parquet data<br/>nodes · edges · transactions] --> V[Validation]
    V --> G[Directed weighted graph]
    G --> F[Graph, temporal and seed features]
    F --> L[Louvain clustering<br/>on documented undirected projection]
    L --> R[Analytics-owned role engine]
    R --> Q[Analytics-owned priority engine]
    Q --> E[Deterministic explanations]
    E --> X[CSV exports<br/>nodes_roles · clusters · top_nodes]
    X --> UI[Streamlit analyst UI]
    G --> UI
    UI --> LLM[Optional grounded LLM assistant]
    LLM --> T[Read-only deterministic graph tools]
    T --> X
    T --> G
```

The interface is intentionally downstream from the final role and priority
engines. It reads their exported scores and explanations; it does not reimplement
or mutate analytic decisions. The optional LLM receives data only through the
read-only deterministic tool layer and is not required by the pipeline or UI.
