#!/bin/bash
DIR=$1
for MODEL in "sentence-transformers/all-MiniLM-L6-v2" "Qwen/Qwen3-Embedding-0.6B"; do
  for DOCS in "10000" "100000" "1000000"; do
    echo "building $MODEL dataset of $DOCS docs"
    python -m hseb.preprocess --queries $DIR/queries.jsonl --corpus $DIR/corpus.jsonl --model $MODEL --corpus-sample $DOCS --batch-size 64
  done
done
