from __future__ import annotations
from pydantic import BaseModel
from hseb.core.dataset import Query
from hseb.core.response import DocScore, SearchResponse
from hseb.core.config import IndexArgs, SearchArgs

import json
from structlog import get_logger

logger = get_logger()


class ExperimentResult(BaseModel):
    tag: str
    indexing_time: list[float]
    index_args: IndexArgs
    search_args: SearchArgs
    queries: list[QueryResult]
    warmup_latencies: list[float]

    @staticmethod
    def from_json(path: str) -> ExperimentResult:
        with open(path, "r") as file:
            raw = json.load(file)
            return ExperimentResult(**raw)

    def to_json(self, workdir: str):
        out_file = f"{workdir}/{self.tag}-{self.index_args.to_string()}-{self.search_args.to_string()}.json"
        # logger.debug(f"Saved experiment result to {out_file}")
        with open(out_file, "w") as file:
            file.write(json.dumps(self.model_dump()))


class QueryResult(BaseModel):
    query_id: int
    exact: list[DocScore]
    response: list[DocScore]
    client_latency: float

    @staticmethod
    def from_response(
        query: Query,
        search_args: SearchArgs,
        response: SearchResponse,
    ) -> QueryResult:
        match search_args.filter_selectivity:
            case 10:
                exact = query.exact10
            case 90:
                exact = query.exact90
            case 100:
                exact = query.exact100
        return QueryResult(
            query_id=query.id,
            exact=exact,
            response=response.results,
            client_latency=response.client_latency,
        )
