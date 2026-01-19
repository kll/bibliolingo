# BiblioLingo

Shhh… I'm indexing. If it's not written down, it didn't happen (I checked).

A Retrieval-Augmented Generation (RAG) pipeline for internal technical documentation using **hybrid search** (BM25 + vector similarity).

## What is this?

BiblioLingo is a single point of entry for dispersed knowledge across engineering systems. It prioritizes "Document First" philosophy by preferring ADRs and GitHub docs over other documentation sources.

**Key Features:**

- **Hybrid Retrieval**: Combines BM25 keyword search with vector semantic search using Reciprocal Rank Fusion (RRF)
- **Pure Relevance Ranking**: Results sorted by relevance scores without source-based boosting
- **Metadata Filtering**: Filter by document type, source, and component tags
- **Smart Chunking**: Markdown-aware splitting with special ADR section detection
- **Answer Generation**: Optional LLM-based answers with inline citations
- **Evaluation Framework**: Recall@k, MRR, and Precision@k metrics

## Architecture

**Tech Stack:**

- **Language**: Python 3.10+
- **Vector Store**: MongoDB Atlas (via Docker Compose)
- **Lexical Search**: rank-bm25
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI (gpt-4o-mini default)
- **Framework**: LangChain components + custom retrieval logic

**Pipeline Components:**

1. **Ingestion**: Loads markdown files, extracts metadata, chunks by headings with ADR-aware splitting
2. **Deduplication**: MinHash-based near-duplicate detection (>85% similar) with tiebreaker by recency/length
3. **Indexing**: Generates embeddings and stores in MongoDB with vector search index
4. **Retrieval**: Hybrid search combining BM25 and vector similarity with RRF fusion
5. **Score Normalization**: Normalizes RRF scores to 0-1 range for confidence checking
6. **Generation**: Optional LLM-based answer synthesis with inline citations

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for MongoDB)
- OpenAI API key

## Setup

1. **Clone the repository**

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenAI API key:

   ```bash
   OPENAI_API_KEY=sk-...
   ```

4. **Start MongoDB**

   ```bash
   docker-compose up -d
   ```

5. **Place your markdown files**

   Use the scripts in `scripts/` to export and organize your documents in the `./data` directory (gitignored):

   ```text
   data/
   ├── confluence/
   │   ├── ADRs/
   │   └── DEV/
   └── github/
       └── docs/
   ```

## Usage

### Ingest Documents

Process and index all markdown files:

```bash
python -m cli.ingest --data-dir ./data
```

This will:

- Load all markdown files from data directory
- Extract metadata and normalize
- Chunk documents by headings (ADR-aware)
- Deduplicate using MinHash
- Generate embeddings via OpenAI
- Store in MongoDB with vector index

**Note:** You may see a warning about creating the vector search index. This is expected with MongoDB Atlas Local. Continue to the next step to create it manually.

### Create Vector Search Index

After ingestion, you need to manually create the vector search index in MongoDB Compass:

1. Open **MongoDB Compass** and connect to `mongodb://localhost:27017/?directConnection=true`
2. Navigate to the **bibliolingo** database → **chunks** collection.
3. Click on the **"Indexes"** tab.
4. Click on **SEARCH INDEXES** viewing mode.
5. Click on the **Create Atlas Search Index** button.
6. Use `vector_index` for the name of the search index.
7. Select **"Vector Search"** for the search index type.
8. Paste this exact configuration:

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

9. Click **"Create Search Index"**

Once created, verify the index appears in the Search Indexes list. The hybrid retrieval system will now work with both BM25 and vector search.

### Query the System

Basic query:

```bash
python -m cli.query "How do I set up feature flags?"
```

With filters:

```bash
python -m cli.query "API gateway selection" --doc-type ADR --k 5
```

Adjust BM25 weight (favor keywords):

```bash
python -m cli.query "YARP configuration" --alpha 0.7
```

Generate LLM answer with citations:

```bash
python -m cli.query "Why did we choose YARP?" --generate-answer
```

### Run Evaluation

Evaluate retrieval performance on the gold dataset:

```bash
python -m cli.evaluate --gold-path eval/gold.jsonl
```

Results are saved with a timestamp: `artifacts/eval_report_YYYYMMDD_HHMMSS.json`

This allows you to track performance over time as you tweak settings. You can also specify a custom output path:

```bash
python -m cli.evaluate --output-path ./artifacts/my_eval.json
```

## Configuration

All settings are configured via environment variables in `.env`:

```bash
# OpenAI API (REQUIRED)
OPENAI_API_KEY=sk-...

# MongoDB connection
MONGO_DB_URL=mongodb://localhost:27017/?directConnection=true
DB_NAME=bibliolingo
COLLECTION_NAME=chunks

# Retrieval settings
DEFAULT_ALPHA=0.5          # BM25 weight (0=vector only, 1=BM25 only)
DEFAULT_TOP_K=10
RRF_K=60                   # Reciprocal rank fusion constant
CONFIDENCE_THRESHOLD=0.3   # Normalized score threshold (0-1) for fallback

# LLM settings
DEFAULT_LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1000

# Logging
LOG_LEVEL=INFO
```

## Document Types

The system automatically detects and classifies documents:

- **ADR**: Architecture Decision Records (highest priority)
- **RFC**: Request for Comments
- **DESIGN**: Design documents
- **HOWTO**: Tutorials and setup guides
- **RUNBOOK**: Operational runbooks
- **POSTMORTEM**: Incident postmortems

## Deduplication

The system uses MinHash-based deduplication during ingestion to detect near-duplicate content:

### When It Activates

- **Threshold**: 85% Jaccard similarity (very similar content)
- **Scope**: Only runs during `python -m cli.ingest` - not during queries
- **Purpose**: Prevents indexing multiple copies of the same content

### How It Works

1. Calculate MinHash signature for each chunk
2. Find clusters of chunks with >85% similarity
3. For each cluster, keep one "canonical" version based on:
   - Most recent `updated_at` timestamp (prefer fresh content)
   - Longest content (prefer complete version)
   - Priority score (ADR=10, GitHub=8, Confluence DEV=5) as final tiebreaker

### Important Notes

- **This does NOT affect retrieval ranking** - queries use pure relevance scores
- Only activates if duplicate content exists across sources
- Acts as a safeguard to reduce index size and prevent redundant results
- Priority selection is just a tiebreaker when timestamps/length are equal

## Evaluation

The gold dataset (`eval/gold.jsonl`) contains 20 test queries with known relevant documents. Evaluation metrics include:

- **Recall@k**: Percentage of queries with at least one relevant doc in top-k
- **MRR**: Mean reciprocal rank of first relevant document
- **Precision@k**: Average precision in top-k results

Target metrics: **Recall@5 > 70%**, **Recall@10 > 85%**

## Confidence Scoring

The system uses **normalized confidence scores** (0-1 range) to assess result quality:

### How It Works

1. **RRF scores** are calculated from BM25 and vector rankings
2. **Scores are normalized** by dividing by the theoretical maximum possible RRF score
3. **Final scores** represent percentage of the best possible result (1.0 = perfect)

### Understanding the Threshold

The default threshold is `0.3`, meaning:

- A score of `0.3` = top result is **30% of the theoretical maximum**
- The maximum (1.0) would require: rank #1 in both BM25 and vector search
- Typical good results score **0.4-0.7**
- Scores below threshold trigger fallback search with relaxed filters

### Example Score Calculation

With default settings (rrf_k=60, alpha=0.5):

- Max possible RRF: `0.5/(60+1) + 0.5/(60+1) = 0.0164`
- A document ranking #1 in both = **1.0 normalized score**
- A document ranking #3 in BM25, #5 in vector = **~0.4 normalized score**

### Tuning the Threshold

Lower threshold (e.g., `0.2`) = fewer false warnings, more lenient
Higher threshold (e.g., `0.5`) = stricter quality control, more fallbacks

Adjust via `.env`: `CONFIDENCE_THRESHOLD=0.3`

## Project Structure

```text
bibliolingo/
├── src/
│   ├── ingestion/          # Document loading, chunking, indexing
│   ├── retrieval/          # Hybrid search, reranking, citations
│   ├── generation/         # LLM answer generation
│   ├── evaluation/         # Metrics calculation
│   └── utils/              # Config, deduplication
├── cli/                    # Command-line interfaces
│   ├── ingest.py
│   ├── query.py
│   └── evaluate.py
├── eval/
│   └── gold.jsonl          # Evaluation dataset
├── data/                   # GITIGNORED - markdown source files
├── artifacts/              # GITIGNORED - embeddings, indexes
├── docker-compose.yml      # MongoDB setup
├── requirements.txt
└── .env.example
```

## License

MIT License - feel free to use this for learning and building!
