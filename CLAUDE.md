# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

BiblioLingo is a prototype Retrieval-Augmented Generation (RAG) pipeline that provides a single point of entry for dispersed knowledge across multiple systems. It implements **hybrid retrieval** (BM25 + vector search) with intelligent source preference for internal technical documentation.

**Key Features:**
- Hybrid search combining BM25 keyword matching with semantic vector search
- Source preference: Prioritizes ADRs and GitHub docs over Confluence DEV docs
- Metadata-filtered retrieval with doc_type, source, and component tags
- MinHash-based deduplication with priority selection
- OpenAI-powered answer generation with inline citations
- Comprehensive evaluation framework (Recall@k, MRR, Precision@k)

## Architecture

### Tech Stack
- **Language:** Python 3.10+
- **Vector Store:** MongoDB Atlas Local (Docker)
- **Lexical Search:** rank-bm25
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** OpenAI gpt-4o-mini
- **Framework:** LangChain + custom retrieval logic

### Module Structure

```
src/
├── ingestion/           # Document loading, parsing, chunking, indexing
│   ├── loader.py        # Loads markdown from data/confluence and data/github
│   ├── normalizer.py    # Extracts metadata, detects doc_type, assigns priority
│   ├── chunker.py       # Markdown-aware chunking with ADR section detection
│   └── indexer.py       # Generates embeddings, stores in MongoDB
├── retrieval/           # Hybrid search and score normalization
│   ├── hybrid_retriever.py      # BM25 + vector with RRF fusion
│   ├── score_normalizer.py      # Score normalization and confidence checking
│   ├── citation_formatter.py    # Format citations without full content
│   └── metadata_filter.py       # (not yet implemented)
├── generation/          # LLM answer synthesis
│   └── answer_generator.py      # OpenAI-based answers with citations
├── evaluation/          # Metrics and evaluation
│   ├── metrics.py       # Recall@k, MRR, Precision@k
│   └── eval_runner.py   # Evaluation harness
└── utils/               # Shared utilities
    ├── config.py        # Environment configuration
    └── deduplicator.py  # MinHash deduplication

cli/                     # Command-line interfaces
├── ingest.py           # Ingestion pipeline
├── query.py            # Query interface
└── evaluate.py         # Evaluation runner
```

## Key Concepts

### Document Types
The system detects and classifies documents automatically:
- **ADR** (Priority 10): Architecture Decision Records - highest priority
- **RFC** (Priority 9): Request for Comments
- **DESIGN** (Priority 8-9): Design documents
- **HOWTO** (Priority 6-8): Tutorials and guides (GitHub preferred)
- **RUNBOOK** (Priority 6-7): Operational runbooks
- **POSTMORTEM** (Priority 6-7): Incident analysis

### Relevance Ranking

Results are ranked purely by relevance scores from hybrid retrieval - no source-based boosting is applied. The system uses:

- RRF (Reciprocal Rank Fusion) to combine BM25 and vector search results
- Score normalization (0-1 range) for confidence checking
- Fallback logic for low confidence results

### Hybrid Retrieval (RRF)
Combines BM25 and vector search using Reciprocal Rank Fusion:
- Formula: `score = sum(weight_i / (k + rank_i))` for each method
- Default alpha: 0.5 (equal weighting)
- RRF constant k: 60

## Common Commands

### Development
```bash
# Start MongoDB
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Run ingestion
python -m cli.ingest --data-dir ./data

# Query (basic)
python -m cli.query "How do feature flags work?"

# Query with filters
python -m cli.query "API gateway" --doc-type ADR --k 5

# Query with answer generation
python -m cli.query "Why YARP?" --generate-answer

# Run evaluation
python -m cli.evaluate --gold-path eval/gold.jsonl
```

## Data Organization

**Data sources (gitignored):**
- `data/confluence/ADRs/` - Architecture Decision Records
- `data/confluence/DEV/` - General development documentation
- `data/github/bl-platform/docs/` - GitHub markdown docs

**Artifacts (gitignored):**
- `artifacts/chunk_metadata.jsonl` - Chunk metadata cache
- `artifacts/bm25_index.pkl` - BM25 index cache
- `artifacts/eval_report_*.json` - Timestamped evaluation reports

## Important Patterns

### Chunking Strategy
- **Standard docs:** Split by markdown headings (h1-h6)
- **ADRs:** Special handling for Context, Decision, Consequences, Alternatives sections
- **Large sections:** Split by paragraphs, then sentences if needed
- **Size limits:** Min 50 chars, max 2000 chars per chunk

### Deduplication

**Purpose:** Prevents indexing duplicate content during ingestion only - does NOT affect retrieval ranking.

**How it works:**

- Uses MinHash with 85% Jaccard similarity threshold
- When duplicates found (>85% similar), keeps one canonical version based on:
  1. Most recent `updated_at` timestamp (prefer fresh content)
  2. Longest content (prefer complete version)
  3. Priority score as final tiebreaker: ADR=10, GitHub=8, Confluence DEV=5

**Important:** This is purely a tiebreaker mechanism during ingestion. Query results are ranked by pure relevance scores, not by source priority.

### Metadata Schema
Each chunk includes:
- `chunk_id`: Unique identifier (e.g., "confluence-3074097170-decision-0")
- `doc_id`: Parent document ID
- `doc_type`: ADR|RFC|DESIGN|HOWTO|RUNBOOK|POSTMORTEM
- `source`: confluence|github
- `source_path`: Relative path for citations
- `section_heading`: Current section
- `section_hierarchy`: Breadcrumb trail
- `section_type`: For ADRs (context|decision|consequences|alternatives)
- `component_tags`: Extracted technology tags
- `priority_score`: Deduplication priority
- `embedding`: 1536-dim vector

## MongoDB Vector Search Index

**IMPORTANT:** The vector search index must be created manually in MongoDB Compass.

Index name: `vector_index`
Configuration:
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "doc_type"
    },
    {
      "type": "filter",
      "path": "source"
    },
    {
      "type": "filter",
      "path": "component_tags"
    }
  ]
}
```

Without the filter fields, queries with `--doc-type` or `--source` filters will fail.

## Public Repo Guidelines

This is a **PUBLIC** repository. Never commit:
- Raw document content in code files
- Full document paths that reveal internal structure
- Company-specific examples in comments
- API keys or credentials
- Files in `data/` or `artifacts/` directories

Always use:
- Generic examples: "How do feature flags work?" not "How does BL handle PII?"
- Relative paths: `./data/` not absolute paths
- Sanitized doc IDs: "confluence-3074097170" not full file content
- Generic logging: "Processing ADR..." not "Processing Boostlingo billing ADR..."

## Testing & Evaluation

### Gold Dataset Format
`eval/gold.jsonl` contains test queries:
```json
{"query": "...", "target_docs": ["doc_id1", "doc_id2"], "expected_doc_type": "ADR"}
```

### Target Metrics
- Recall@5: > 70%
- Recall@10: > 85%
- MRR: Higher is better (1.0 = perfect first-rank retrieval)

## Troubleshooting

### Vector Search Errors
If you see "Path 'doc_type' needs to be indexed as filter":
- Delete and recreate the vector_index in MongoDB Compass
- Ensure all filter fields are included (doc_type, source, component_tags)

### Low Confidence Scores

**Understanding Confidence Scoring:**

- Scores are **normalized to 0-1 range** where 1.0 is the theoretical maximum
- The maximum requires: rank #1 in both BM25 and vector search
- Default threshold: 0.3 (30% of theoretical maximum)
- Typical good results: 0.4-0.7 normalized score

**How Normalization Works:**

1. Calculate max possible RRF score (rank #1 in both searches)
2. Divide all RRF scores by this maximum
3. Result: intuitive 0-1 scale where 0.3 threshold means "30% as good as best possible"

**Automatic Fallback:**

The system falls back to relaxed filters when confidence < threshold:

1. First retry: Remove doc_type filter (keep source)
2. Second retry: Remove all filters

**Tuning:**

- Lower threshold (0.2): fewer warnings, more lenient
- Higher threshold (0.5): stricter quality control
- Adjust via `.env`: `CONFIDENCE_THRESHOLD=0.3`

### BM25-Only Mode
If vector search is unavailable, use BM25-only:
```bash
python -m cli.query "your query" --alpha 0.9
```
