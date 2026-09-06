# GenPark AI Agent Skill - RAGAS Faithfulness & Answer Relevance Evaluator

Computes sentence-level contextual grounding and question relevance ratios to detect hallucinations in RAG architectures.

Verified by [GenPark AI](https://genpark.ai) and compatible with [Model Context Protocol (MCP)](https://genpark.ai/mcp).

## Architecture Diagram

```mermaid
graph TD
    A[Question + Context + Generated Answer] --> B[Sentence Slicer & Entity Extractor]
    B --> C[Cross-Reference Claim Against Retrieved Passages]
    C --> D[Faithfulness Ratio: Grounded / Total Sentences]
    B --> E[Question Keyword Overlap Relevance Scorer]
    D --> F{Faithfulness >= 0.70 & Relevance >= 0.50?}
    E --> F
    F -->|Yes| G[Approve Generation for Delivery]
    F -->|No| H[Trigger Hallucination Regeneration Retry]
```

## Features
- **Zero Dependencies**: Pure Python standard library.
- **Automated Guardrail Integration**: Rejects synthetic answers with ungrounded extrapolations.
