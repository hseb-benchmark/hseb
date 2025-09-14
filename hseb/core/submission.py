from __future__ import annotations
import datetime
import json
from pydantic import BaseModel, Field
from hseb.core.config import Config, IndexArgs, SearchArgs
from tqdm import tqdm
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from hseb.core.measurement import ExperimentResult
from structlog import get_logger

logger = get_logger()


class QueryMetrics(BaseModel):
    recall5: list[float] = Field(default_factory=list)
    recall10: list[float] = Field(default_factory=list)
    recall30: list[float] = Field(default_factory=list)
    latency: list[float] = Field(default_factory=list)


class ExperimentMetrics(BaseModel):
    tag: str
    indexing_time: list[float]
    index_args: IndexArgs
    search_args: SearchArgs
    metrics: QueryMetrics
    warmup_latencies: list[float]

    @staticmethod
    def recall_score(true_docs: list[int], retrieved_docs: list[int], k: int) -> float:
        k_adj = min(k, len(retrieved_docs), len(true_docs))
        retrieved_set = set(retrieved_docs[:k_adj])
        true_set = set(true_docs[:k_adj])
        hits = len(retrieved_set.intersection(true_set))
        return hits / float(len(true_set))

    @staticmethod
    def from_experiment(exp: ExperimentResult) -> ExperimentMetrics:
        metrics = QueryMetrics()
        for query in exp.queries:
            ground_truth = [doc.doc for doc in query.exact]
            retrieved_docs = [doc.doc for doc in query.response]
            metrics.recall5.append(ExperimentMetrics.recall_score(ground_truth, retrieved_docs, 5))
            metrics.recall10.append(ExperimentMetrics.recall_score(ground_truth, retrieved_docs, 10))
            metrics.recall30.append(ExperimentMetrics.recall_score(ground_truth, retrieved_docs, 30))
            metrics.latency.append(query.client_latency)
        return ExperimentMetrics(
            tag=exp.tag,
            indexing_time=exp.indexing_time,
            index_args=exp.index_args,
            search_args=exp.search_args,
            metrics=metrics,
            warmup_latencies=exp.warmup_latencies,
        )


class Submission(BaseModel):
    time: str
    version: str
    config: Config
    experiments: list[ExperimentMetrics]

    @staticmethod
    def from_dir(config: Config, path: str) -> Submission:
        experiments = []
        for file in tqdm(list(Path(path).iterdir()), desc="loading measurements"):
            if file.is_file() and file.name.endswith(".json"):
                exp = ExperimentResult.from_json(file)
                experiments.append(ExperimentMetrics.from_experiment(exp))
        try:
            hseb_version = version("hseb")
        except PackageNotFoundError:
            hseb_version = "unknown"
        logger.info(f"Loaded {len(experiments)} experiments")
        return Submission(
            time=datetime.datetime.now().isoformat(),
            version=hseb_version,
            config=config,
            experiments=experiments,
        )

    @staticmethod
    def from_json(path: str) -> Submission:
        with open(path, "r") as file:
            raw = json.loads(file.read())
            return Submission(**raw)

    def to_json(self, path: str):
        if Path(path).exists():
            raise Exception(f"output file {path} already exists")
        with open(path, "w") as file:
            file.write(json.dumps(self.model_dump()))
