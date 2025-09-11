# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hseb is a vector search benchmarking tool designed to evaluate the performance of different search engines (Nixiesearch, Qdrant, Elastic). It focuses on measuring search latency and recall accuracy for semantic search operations.

## Development Commands

### Python Environment Setup
```bash
pip install -e .[test]  # Install with test dependencies
```

### Testing
```bash
pytest                  # Run all tests
pytest --skip-slow     # Skip slow-running tests
```

### Code Quality
```bash
ruff check             # Run linting
ruff format            # Format code
```

### Benchmarking Operations
```bash
# Run full benchmark with config file
python -m hseb --config configs/nixiesearch/dev.yml --out results.json

# Optional: cleanup containers after benchmark
python -m hseb --config configs/nixiesearch/dev.yml --out results.json --delete-container true
```

## Architecture

### Core Components

- **hseb/core/**: Core benchmarking infrastructure
  - `config.py`: YAML-based experiment configuration with parameter matrices
  - `dataset.py`: HuggingFace dataset loading and batching for corpus/queries
  - `measurement.py`: Result collection (ExperimentResult, Submission classes)
  - `response.py`: Search result wrapper with latency tracking
  - `stats.py`: Statistical analysis utilities

- **hseb/engine/**: Search engine abstraction layer
  - `base.py`: Abstract base class with start(), stop(), index_batch(), search() interface
  - `nixiesearch/nixiesearch.py`: Nixiesearch implementation using Docker containers
  - `qdrant/qdrant.py`: Qdrant implementation 
  - `elastic/elastic.py`: Elasticsearch implementation

- **hseb/__main__.py**: Main benchmarking execution orchestrating index + search phases
- **hseb/preprocess.py**: Data preprocessing utilities

### Benchmarking Workflow

The main benchmark (`__main__.py`) follows this execution pattern:
1. **Configuration Loading**: Loads YAML config defining engine, dataset, and experiment parameters
2. **Parameter Matrix Expansion**: Generates all combinations from IndexArgsMatrix and SearchArgsMatrix
3. **Index Phase**: For each index configuration, starts engine container, indexes corpus in batches, commits
4. **Search Phase**: For each search configuration, runs warmup queries then measures all test queries
5. **Result Collection**: Gathers measurements into ExperimentResult objects and final Submission

### Configuration System

Experiments are defined in YAML files (see `configs/nixiesearch/dev.yml`) with:
- Engine selection via fully-qualified class name (e.g., `hseb.engine.nixiesearch.nixiesearch.NixiesearchEngine`)
- Docker image specification for containerized engines
- Dataset configuration (HuggingFace dataset name, embedding dimension)
- Parameter matrices for systematic benchmarking:
  - `IndexArgsMatrix`: m, ef_construction, quantization, batch_size, plus engine-specific kwargs
  - `SearchArgsMatrix`: ef_search, filter_selectivity, plus engine-specific kwargs

### Search Engine Integration

Each engine implementation inherits from `EngineBase` and implements:
- `start(index_args)`: Start containerized engine (uses Docker Python client)  
- `stop()`: Stop and cleanup engine container
- `index_batch(batch, index_args)`: Index a batch of documents with embeddings
- `commit()`: Finalize indexing (flush, merge segments)
- `search(search_args, query, top_k)`: Execute vector search, return Response with results and latency

Engine instances are dynamically loaded via `EngineBase.load_class()` from config.engine string.

### Dataset Schema

Uses HuggingFace datasets with predefined schemas defined in `dataset.py`:

**CORPUS_SCHEMA**: Documents with id (int32), text (string), embedding (float32[]), tag (int32[])
**QUERY_SCHEMA**: Queries with same fields plus ground truth results for recall calculation:
- results_10_docs/scores, results_90_docs/scores, results_100_docs/scores

The `BenchmarkDataset` class loads corpus/query datasets and provides:
- `corpus_batched(batch_size)`: Generator yielding Doc batches for indexing
- `queries()`: Generator yielding Query objects with ground truth for evaluation

## Key Dependencies

- `datasets`: HuggingFace dataset loading and processing
- `docker`: Container management for search engines
- `pydantic`: Configuration validation and data models
- `structlog`: Structured logging throughout the benchmark
- `qdrant-client`, `elasticsearch`: Client libraries for respective search engines
- `faiss-cpu`: Vector similarity operations for recall calculations
- `sentence_transformers`: Not directly used (embeddings come from dataset)
- `typed-argparse`: CLI parsing