from dataclasses import dataclass


@dataclass
class Response:
    results: list[int]
    scores: list[float]
    client_latency: float
    server_latency: float
