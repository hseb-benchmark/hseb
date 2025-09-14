#!/bin/bash

#for MODEL in "sentence-transformers/all-MiniLM-L6-v2" "intfloat/e5-base-v2" "Qwen/Qwen3-Embedding-4B"; do
for MODEL in "Qwen/Qwen3-Embedding-4B"; do
  for SIZE in 1000 100000 1000000; do
    python -m hseb.preprocess --queries ~/data/beir/msmarco/queries.jsonl --corpus ~/data/beir/msmarco/corpus.jsonl --model $MODEL --out ~/data/hseb-out --corpus-sample $SIZE --batch-size 16
  done
done