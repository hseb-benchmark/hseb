from dataclasses import dataclass


@dataclass
class DocScore:
    doc: int
    score: float


@dataclass
class Response:
    results: list[DocScore]
    client_latency: float
