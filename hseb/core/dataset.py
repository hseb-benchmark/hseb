from __future__ import annotations
from hseb.core.config import DatasetConfig
from datasets import load_dataset, Features, Value, Sequence, Dataset
from dataclasses import dataclass
import numpy as np
from typing import Generator, Any
from hseb.core.response import DocScore
from structlog import get_logger

logger = get_logger()

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
    query_dataset: Dataset
    corpus_dataset: Dataset

    def __init__(self, config: DatasetConfig) -> None:
        self.query_dataset = load_dataset(config.name, config.query, split="train", features=QUERY_SCHEMA)
        self.corpus_dataset = load_dataset(config.name, config.corpus, split="train", features=CORPUS_SCHEMA)
        logger.info(f"Loaded dataset: {len(self.query_dataset)} queries, {len(self.corpus_dataset)} documents")

    def corpus_batched(self, batch_size: int) -> Generator[list[Doc], Any, None]:
        for batch in self.corpus_dataset.batch(batch_size):
            result: list[Doc] = []
            for id, text, embedding, tag in zip(batch["id"], batch["text"], batch["embedding"], batch["tag"]):
                result.append(
                    Doc(
                        id=id,
                        text=text,
                        embedding=np.array(embedding),
                        tag=tag,
                    )
                )
            yield result

    def corpus(self) -> Generator[Doc, Any, None]:
        for id, text, embedding, tag in zip(
            self.corpus_dataset["id"],
            self.corpus_dataset["text"],
            self.corpus_dataset["embedding"],
            self.corpus_dataset["tag"],
        ):
            yield Doc(
                id=id,
                text=text,
                embedding=np.array(embedding),
                tag=tag,
            )

    def queries(self, limit: int | None = None) -> Generator[Query, Any, None]:
        if limit is None:
            for row in self.query_dataset:
                yield Query.from_dict(row)
        else:
            for row in self.query_dataset.take(limit):
                yield Query.from_dict(row)


@dataclass
class Query:
    id: int
    text: str
    embedding: np.ndarray
    exact10: list[DocScore]
    exact90: list[DocScore]
    exact100: list[DocScore]

    @staticmethod
    def from_dict(json: dict) -> Query:
        return Query(
            id=json["id"],
            text=json["text"],
            embedding=np.array(json["embedding"]),
            exact10=[DocScore(doc, score) for doc, score in zip(json["results_10_docs"], json["results_10_scores"])],
            exact90=[DocScore(doc, score) for doc, score in zip(json["results_90_docs"], json["results_90_scores"])],
            exact100=[
                DocScore(doc, score) for doc, score in zip(json["results_100_docs"], json["results_100_scores"])
            ],
        )


@dataclass
class Doc:
    id: int
    text: str
    embedding: np.ndarray
    tag: list[int]
