from __future__ import annotations
from hseb.core.response import DocScore, Response
from hseb.core.config import Config, IndexArgs, SearchArgs
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    tag: str
    index_args: IndexArgs
    search_args: SearchArgs
    measurements: list[Measurement]


@dataclass
class Measurement:
    query_id: int
    exact: list[DocScore]
    response: list[DocScore]
    client_latency: float

    @staticmethod
    def from_response(
        query_id: int,
        exact: list[DocScore],
        response: Response,
    ) -> Measurement:
        return Measurement(
            query_id=query_id,
            exact=exact,
            response=response.results,
            client_latency=response.client_latency,
        )


@dataclass
class Submission:
    time: str
    version: str
    config: Config
    experiments: list[ExperimentResult]
