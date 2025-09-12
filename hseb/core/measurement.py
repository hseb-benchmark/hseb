from __future__ import annotations
from pydantic import BaseModel
from hseb.core.response import DocScore, Response
from hseb.core.config import Config, IndexArgs, SearchArgs
from structlog import get_logger
from pathlib import Path
import json
from tqdm import tqdm
from importlib.metadata import PackageNotFoundError, version
import datetime

logger = get_logger()


class ExperimentResult(BaseModel):
    tag: str
    indexing_time: float
    index_args: IndexArgs
    search_args: SearchArgs
    measurements: list[Measurement]
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


class Measurement(BaseModel):
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


class Submission(BaseModel):
    time: str
    version: str
    config: Config
    experiments: list[ExperimentResult]

    @staticmethod
    def from_dir(config: Config, path: str) -> Submission:
        experiments = []
        for file in tqdm(list(Path(path).iterdir()), desc="loading measurements"):
            if file.is_file() and file.name.endswith(".json"):
                experiments.append(ExperimentResult.from_json(file))
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
