# HSEB: Hybrid Search Engine Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/hseb-benchmark/hseb)
![Last commit](https://img.shields.io/github/last-commit/hseb-benchmark/hseb)
![Last release](https://img.shields.io/github/release/hseb-benchmark/hseb)

HSEB benchmarks search engines to help you pick the right one for your use case. Right now it focuses on vector search, but we're planning to add sparse and lexical search too.

## Use Cases

### Research & Academia
Compare different algorithms and publish reproducible results. HSEB runs everything in Docker containers so your benchmarks work the same way on different machines. You can test how HNSW parameters affect search quality, compare float32 vs int8 vs binary vectors, and generate performance charts for papers.

### Industry & Production
Figure out which search engine to use and how to configure it. HSEB helps with vendor selection, capacity planning, and finding the best settings for your workload. You can also analyze costs and performance trade-offs.

### Algorithm Development
Add your own search engine implementation and test it against the competition. The configuration system makes it easy to test different parameter combinations and see how filtering affects performance.

## Supported Engines

| Engine | Version | Container | Quantization |
|--------|---------|-----------|--------------|
| **Nixiesearch** | 0.6.x | Docker | float32, int8, binary |
| **Qdrant** | 1.x | Docker | float32, int8, binary |
| **Elasticsearch** | 8.x, 9.x | Docker | float32, int8, binary |
| **OpenSearch** | 2.x, 3.x | Docker | float32, int8, binary |

## Quick Start

```bash
# Install
pip install -e .[test]

# Run a benchmark
python -m hseb --config configs/qdrant/dev.yml --out results.json

# Clean up containers afterward
python -m hseb --config configs/opensearch/dev.yml --out results.json --delete-container true
```

### Example Configuration

```yaml
engine: hseb.engine.qdrant.qdrant.QdrantEngine
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

1. Create `hseb/engine/yourengine/` directory
2. Implement the `EngineBase` interface in `yourengine.py`
3. Add config file at `configs/yourengine/dev.yml`
4. Add dependencies to `pyproject.toml`

Your engine needs these methods:
- `start(index_args)` - Start the containerized engine
- `index_batch(batch, index_args)` - Index a batch of documents
- `commit()` - Finish indexing
- `search(search_args, query, top_k)` - Run a vector search
- `stop()` - Clean up

## Requirements

- Python 3.11+
- Docker (for running search engines)
- 8GB+ RAM (16GB recommended for large datasets)
- Storage varies by dataset (usually 1-10GB per experiment)

## License

Apache 2.0 License - see LICENSE file for details.

## Contributing

Pull requests welcome! Just make sure tests pass and code is formatted properly. If you're adding a new engine, include tests and benchmark results.