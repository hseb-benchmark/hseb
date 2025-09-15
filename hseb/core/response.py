from dataclasses import dataclass


@dataclass
class DocScore:
    doc: int
    score: float


@dataclass
class SearchResponse:
    results: list[DocScore]
    client_latency: float


@dataclass
class IndexResponse:
    client_latency: float
