from __future__ import annotations
from hseb.core.response import Response
from typing import Any
import numpy as np
from hseb.core.config import IndexArgs, SearchArgs
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    tag: str
    index_args: IndexArgs
    search_args: SearchArgs
    measurements: list[Measurement]


@dataclass
class DocList:
    docs: list[int]
    scores: list[float]


@dataclass
class Measurement:
    query_id: int
    exact: DocList
    response: DocList
    server_latency: float
    client_latency: float

    @staticmethod
    def from_response(
        query_id: int,
        exact: DocList,
        response: Response,
    ) -> Measurement:
        return Measurement(
            query_id=query_id,
            exact=exact,
            response=DocList(docs=response.results, scores=response.scores),
            client_latency=response.client_latency,
            server_latency=response.server_latency,
        )
