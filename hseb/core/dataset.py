from __future__ import annotations
from hseb.core.config import DatasetConfig
from datasets import load_dataset, Features, Value, Sequence, Dataset
from dataclasses import dataclass
import numpy as np
from torch.utils.data import DataLoader
from typing import Generator, Any
import itertools

CORPUS_SCHEMA = Features(
    {
        "id": Value("int32"),
        "text": Value("string"),
        "embedding": Sequence(Value("float32")),
        "tag": Sequence(Value("int32")),
    }
)

QUERY_SCHEMA = Features(
    {
        "id": Value("int32"),
        "text": Value("string"),
        "embedding": Sequence(Value("float32")),
        "results_10_docs": Sequence(Value("int32")),
        "results_10_scores": Sequence(Value("float32")),
        "results_90_docs": Sequence(Value("int32")),
        "results_90_scores": Sequence(Value("float32")),
        "results_100_docs": Sequence(Value("int32")),
        "results_100_scores": Sequence(Value("float32")),
    }
)


class BenchmarkDataset:
    query: Dataset
    corpus: Dataset

    def __init__(self, config: DatasetConfig) -> None:
        self.query = load_dataset(config.name, config.query, split="train", features=QUERY_SCHEMA)
        self.corpus = load_dataset(config.name, config.corpus, split="train", features=CORPUS_SCHEMA)

    def corpus_batches(self, batch_size: int) -> Generator[list[Doc], Any, None]:
        for batch in self.corpus.batch(batch_size):
            result: list[Doc] = []
            for id, text, embedding, tag in zip(batch["id"], "text", "embedding", "tag"):
                result.append(
                    Doc(
                        id=id,
                        text=text,
                        embedding=np.array(embedding),
                        tag=tag,
                    )
                )
            yield result


@dataclass
class DocScores:
    docs: np.ndarray
    scores: np.ndarray


@dataclass
class Query:
    id: int
    text: str
    embedding: np.ndarray
    exact10: DocScores
    exact90: DocScores
    exact100: DocScores

    @staticmethod
    def from_dict(json: dict) -> Query:
        return Query(
            id=json["id"],
            text=json["text"],
            embedding=json["embedding"],
            exact10=DocScores(
                docs=np.array(json["results_10_docs"]),
                scores=np.array(json["results_10_scores"]),
            ),
            exact90=DocScores(
                docs=np.array(json["results_90_docs"]),
                scores=np.array(json["results_90_scores"]),
            ),
            exact100=DocScores(
                docs=np.array(json["results_100_docs"]),
                scores=np.array(json["results_100_scores"]),
            ),
        )


@dataclass
class Doc:
    id: int
    text: str
    embedding: np.ndarray
    tag: list[int]
