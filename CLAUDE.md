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
python -m hseb.index --corpus path/to/corpus.json --engine nixiesearch
python -m hseb.search --engine nixiesearch --queries path/to/queries.json
```

## Architecture

### Core Components

- **hseb/core/**: Configuration management and response models
  - `config.py`: YAML-based experiment configuration (quantization, HNSW parameters)
  - `response.py`: Search result response wrapper

- **hseb/engine/**: Search engine abstraction layer
  - `base.py`: Abstract base class defining index() and search() interface
  - `nixiesearch.py`: Nixiesearch implementation using Docker containers via docker python client
  - `__init__.py`: Engine factory for loading different backends

- **hseb/**: Main execution scripts
  - `index.py`: Corpus indexing with embeddings and metadata
  - `search.py`: Query execution and recall calculation
  - `preprocess.py`: Data preprocessing utilities

### Configuration System

Experiments are defined in YAML files (see `configs/nixiesearch.yml`) with:
- Engine selection (nixiesearch, qdrant, elastic)
- Batch size for indexing operations
- HNSW parameters: quantization type, M parameter, efConstruction, efSearch

### Search Engine Integration

Each engine implementation:
- Inherits from `EngineBase`
- Implements containerized deployment (Nixiesearch uses Docker via docker python client)
- Handles batch indexing with configurable batch sizes
- Returns structured `Response` objects with latency metrics

### Dataset Schema

Uses HuggingFace datasets with predefined features:
- Documents: id, text, embedding (float32[]), tag (int32[])  
- Queries: same fields plus results_N for different recall@N ground truth

## Key Dependencies

- `sentence_transformers`: Embedding model inference
- `datasets`: HuggingFace dataset handling
- `docker`: Docker container management for search engines
- `faiss-cpu`: Vector similarity operations
- `typed-argparse`: CLI argument parsing