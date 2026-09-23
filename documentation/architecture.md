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
    E --> X[Strict CSV exports<br/>six-field nodes_roles · clusters · top_nodes]
    E --> FOUT[Auxiliary node_features.parquet<br/>metrics · contribution values · explanations]
    P --> M[run_metadata.json<br/>input and output SHA-256 · versions · runtime]
    X --> M
    FOUT --> M
    X --> JOIN[Exact gid one-to-one join<br/>CSV decisions authoritative]
    FOUT --> JOIN
    M --> CHECK[Provenance and freshness verification]
    JOIN --> CHECK
    CHECK --> UI[Streamlit analyst UI<br/>offline embedded assets]
    X --> S[submission/<br/>three validated CSVs and release manifest]
    G --> UI
    UI --> LLM[Optional grounded LLM assistant]
    LLM --> T[Read-only deterministic graph tools]
    T --> JOIN
    T --> G
```

The interface is intentionally downstream from the final role and priority
engines. It reads their exported scores and explanations; it does not reimplement
or mutate analytic decisions. The optional LLM receives data only through the
read-only deterministic tool layer and is not required by the pipeline or UI.
