# HSEB: Hybrid Search Engine Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/hseb-benchmark/hseb)
![Last commit](https://img.shields.io/github/last-commit/hseb-benchmark/hseb)
![Last release](https://img.shields.io/github/release/hseb-benchmark/hseb)

HSEB is a fair search engine benchmarking tool that helps you pick the right engine for your use case. Unlike traditional benchmarks that use fixed parameters, HSEB recognizes that each engine has unique optimal configurations. Some engines benefit from specific HNSW parameters (`m`, `ef_construction`), while others have additional tuning options like segment sizes and memory allocation. 

Instead of comparing raw recall-QPS curves, HSEB finds the Pareto front of optimal configurations per engine, discovering the best QPS each engine can achieve for a given recall level through exhaustive parameter space exploration.

## Features

- **Multi-Engine Support**: Benchmarks Nixiesearch, Qdrant, Elasticsearch, OpenSearch, PostgreSQL + pgvector, Redis, and Weaviate
- **Docker Containerization**: Reproducible benchmarks across environments
- **Vector Quantization**: Compare float32, float16, int8, and binary performance
- **Parameter Testing**: Systematic HNSW configuration evaluation
- **Filter Benchmarks**: Test search performance with different selectivity levels
- **Latency & Recall**: Track both speed and result quality metrics
- **Custom Datasets**: Process your own data with any embedding model
- **Extensible**: Easy to add new search engines

## Supported Engines

| Engine | Version | Index Parameters | Search Parameters |
|--------|---------|------------------|-------------------|
| **Nixiesearch** | 0.6.x | `m`, `ef_construction`, `quant`, `docs_per_segment`, `heap_size`, `ram_buffer_size` | `ef_search`, `filter_selectivity` |
| **Qdrant** | 1.x | `m`, `ef_construction`, `quant`, `max_segment_size_kb`, `original_vectors_on_disk`, `hnsw_on_disk` | `ef_search`, `filter_selectivity` |
| **Elasticsearch** | 8.x, 9.x | `m`, `ef_construction`, `quant`, `docs_per_segment`, `max_merged_segment`, `heap_size` | `ef_search`, `filter_selectivity` |
| **OpenSearch** | 2.x, 3.x | `m`, `ef_construction`, `quant`, `docs_per_segment`, `max_merged_segment`, `heap_size` | `ef_search`, `filter_selectivity` |
| **PostgreSQL + pgvector** | 0.8.x | `m`, `ef_construction`, `quant`, `shared_buffers`, `work_mem`, `maintenance_work_mem` | `ef_search`, `filter_selectivity` |
| **Redis** | 8.x | `m`, `ef_construction`, `quant`, `maxmemory`, `maxmemory_policy` | `ef_search`, `filter_selectivity` |
| **Weaviate** | 1.x | `m`, `ef_construction`, `quant` | `ef_search`, `filter_selectivity` |

## Quick Start

```bash
# Install
pip install hseb

# Download a config file from the repository
wget https://raw.githubusercontent.com/hseb-benchmark/hseb/main/configs/qdrant/dev.yml

# Run a benchmark
python -m hseb --config dev.yml --out results.json
```

You can choose from any of the [available configs](https://github.com/hseb-benchmark/hseb/tree/main/configs) for different engines (qdrant, elastic, nixiesearch, etc).

## Development Installation

```bash
# Clone and install for development
git clone https://github.com/hseb-benchmark/hseb.git
cd hseb
pip install -e .[test]

# Run a benchmark
python -m hseb --config configs/qdrant/dev.yml --out results.json

```

### Example Configuration

```yaml
engine: hseb.engine.qdrant.QdrantEngine
image: qdrant/qdrant:v1.12.5
dataset:
  dim: 384
  name: hseb-benchmark/msmarco
  query: "query-all-MiniLM-L6-v2-100K"
  corpus: "corpus-all-MiniLM-L6-v2-100K"

experiments:
- tag: hnsw-optimization
  k: 100
  index:
    m: [8, 16, 32]
    ef_construction: [64, 128, 256]
    quant: ["float32", "int8"]
  search:
    ef_search: [128, 256, 512]
    filter_selectivity: [10, 90, 100]
```

## Datasets

HSEB uses the MS MARCO dataset as its primary benchmark corpus. We preprocess the data to create embeddings and ground truth results for reproducible evaluation.

### Data Processing Pipeline

The `preprocess.py` script converts MS MARCO data (or your own data) into the format HSEB needs:

```bash
python preprocess.py \
  --queries queries.json \
  --corpus corpus.json \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --queries-sample 1000 \
  --corpus-sample 100000 \
  --out datasets/
```

This process:
1. Takes queries and documents in TREC JSON format (`{"text": "your content"}`)
2. Generates embeddings using any sentence-transformers model
3. Builds a FAISS index for exact nearest neighbor search
4. Assigns filtering tags: 10% of documents get tag "10", 90% get tag "90" (for selectivity benchmarks)
5. Computes ground truth results for different selectivity levels (10%, 90%, 100%)

### Dataset Schema

**Corpus Documents:**
```json
{
  "id": 123,
  "text": "Document content...",
  "embedding": [0.1, 0.2, ...],
  "tag": [10, 90, 100]
}
```

**Query Documents:**
```json
{
  "id": 456,
  "text": "Query text...",
  "embedding": [0.3, 0.4, ...],
  "results_10_docs": [123, 789, ...],
  "results_10_scores": [0.95, 0.87, ...],
  "results_90_docs": [123, 456, ...],
  "results_90_scores": [0.95, 0.89, ...],
  "results_100_docs": [123, 456, ...],
  "results_100_scores": [0.95, 0.89, ...]
}
```

### Using Custom Datasets

To benchmark with your own data:

1. Format your data as TREC JSON files:
```json
{"text": "First document"}
{"text": "Second document"}
```

2. Run preprocessing:
```bash
python preprocess.py \
  --queries my_queries.json \
  --corpus my_corpus.json \
  --model your-preferred-model \
  --out my_dataset/
```

3. Upload to HuggingFace Hub or use locally in your config:
```yaml
dataset:
  dim: 384
  name: your-username/your-dataset
  query: "queries"
  corpus: "corpus"
```

## Development

### Setup

```bash
pip install -e .[test]

# Run tests
pytest                  # All tests
pytest --skip-slow     # Skip slow integration tests

# Code quality
ruff check             # Lint code
ruff format            # Format code
```

### Adding New Engines

Create a new engine by inheriting from `EngineBase`:

```python
# my_custom_engine.py
import docker
import time
from hseb.engine.base import EngineBase
from hseb.core.config import Config, IndexArgs, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import Response, DocScore

class MyCustomEngine(EngineBase):
    def __init__(self, config: Config):
        self.config = config
        self.container = None
        self.client = None

    def start(self, index_args: IndexArgs):
        # Start your engine's Docker container
        docker_client = docker.from_env()
        self.container = docker_client.containers.run(
            image=self.config.image,
            ports={"9200/tcp": 9200},  # Adjust ports as needed
            detach=True,
        )
        
        # Wait for engine to be ready
        self._wait_for_logs(self.container, "started")
        
        # Initialize your client
        # self.client = YourEngineClient(host="localhost", port=9200)
        
        # Create index/collection with HNSW parameters
        # self.client.create_index(...)
        

    def index_batch(self, batch: list[Doc]):
        # Index a batch of documents
        for doc in batch:
            # self.client.index_document(...)

    def commit(self):
        # Finalize indexing (flush, optimize, etc.)
        # self.client.refresh_index("benchmark")

    def search(self, search_args: SearchArgs, query: Query, top_k: int) -> Response:
        
        # Build filter for selectivity testing
        filter_query = None
        if search_args.filter_selectivity != 100:
            # filter_query = {"term": {"tag": search_args.filter_selectivity}}
            pass
        
        start_time = time.time_ns()

        # Execute vector search
        # results = self.client.search(...)
        
        end_time = time.time_ns()
        
        # Convert results to hseb format
        doc_scores = []  # [DocScore(hit.id, hit.score) for hit in results]
        
        return Response(
            results=doc_scores,
            client_latency=(end_time - start_time) / 1000000000.0
        )

    def stop(self, cleanup: bool):
        # Clean up container
        if self.container:
            self.container.stop()
        if cleanup:
            self.container.remove()
```

**Example config file (my_engine_config.yml):**

```yaml
engine: my_custom_engine.MyCustomEngine
image: my-search-engine:latest
dataset:
  dim: 384
  name: hseb-benchmark/msmarco
  query: "query-all-MiniLM-L6-v2-100K"
  corpus: "corpus-all-MiniLM-L6-v2-100K"

experiments:
- tag: test
  k: 100
  index:
    m: [16, 32]
    ef_construction: [128, 256]
    quant: ["float32"]
  search:
    ef_search: [128, 256]
    filter_selectivity: [10, 90, 100]
```

**Install engine-specific dependencies:**

```bash
pip install your-engine-client-library
python -m hseb --config my_engine_config.yml --out results.json
```

## Requirements

- Python 3.12+
- Docker (for running search engines)
- 8GB+ RAM (16GB recommended for large datasets)
- Storage varies by dataset (usually 1-10GB per experiment)

## License

Apache 2.0 License - see LICENSE file for details.

## Contributing

Pull requests welcome! Just make sure tests pass and code is formatted properly. If you're adding a new engine, include tests and benchmark results.