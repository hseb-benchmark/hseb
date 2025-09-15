from __future__ import annotations
import yaml
from typing import Any
from itertools import product
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict
from structlog import get_logger
from enum import StrEnum

logger = get_logger()


class Config(BaseModel):
    engine: str = Field(min_length=1, description="engine type")
    image: str = Field(min_length=1, description="docker image to run")
    dataset: DatasetConfig
    experiments: list[ExperimentConfig]

    @staticmethod
    def from_yaml(text: str) -> Config:
        parsed: dict = yaml.safe_load(text)
        return Config(**parsed)

    @staticmethod
    def from_file(path: str) -> Config:
        with open(path, "r") as f:
            config = Config.from_yaml(f.read())
            logger.info(f"Loaded config file from {path}, engine: {config.engine}")
            return config


class DatasetConfig(BaseModel):
    dim: int = Field(gt=0)
    name: str
    query: str = Field(description="queries dataset name")
    corpus: str = Field(description="corpus dataset name")


class ExperimentConfig(BaseModel):
    tag: str
    k: int
    index: IndexArgsMatrix
    search: SearchArgsMatrix


@dataclass
class IndexArgs:
    m: int
    ef_construction: int
    quant: QuantDatatype
    batch_size: int
    segments: int | None  # none means default
    kwargs: dict[str, Any]

    def to_string(self) -> str:
        parts1 = [f"{key}={value}" for key, value in asdict(self).items() if key != "kwargs"]
        parts2 = [f"{key}={value}" for key, value in self.kwargs.items()]
        return "_".join(parts1 + parts2)


class IndexArgsMatrix(BaseModel):
    m: list[int]
    ef_construction: list[int]
    quant: list[QuantDatatype]
    batch_size: int = 1024
    segments: list[int] | None = Field(default=None)
    kwargs: dict[str, list] = Field(default_factory=dict)

    def expand(self) -> list[IndexArgs]:
        """Generate all permutations of parameters from IndexArgs."""
        segments_list = [None] if self.segments is None or len(self.segments) == 0 else self.segments
        base_params = product(self.m, self.ef_construction, self.quant, segments_list)
        kwargs_combos = product(*self.kwargs.values()) if self.kwargs else [()]

        return [
            IndexArgs(
                m=m,
                ef_construction=ef,
                quant=quant,
                batch_size=self.batch_size,
                segments=segments,
                kwargs=dict(zip(self.kwargs.keys(), kwarg_vals)),
            )
            for (m, ef, quant, segments), kwarg_vals in product(base_params, kwargs_combos)
        ]


class QuantDatatype(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    INT1 = "int1"


@dataclass
class SearchArgs:
    ef_search: int
    filter_selectivity: int
    kwargs: dict[str, Any]

    def to_string(self) -> str:
        parts1 = [f"{key}={value}" for key, value in asdict(self).items() if key != "kwargs"]
        parts2 = [f"{key}={value}" for key, value in self.kwargs.items()]
        return "_".join(parts1 + parts2)


class SearchArgsMatrix(BaseModel):
    ef_search: list[int]
    filter_selectivity: list[int]
    kwargs: dict[str, list] = Field(default_factory=dict)

    def expand(self) -> list[SearchArgs]:
        """Generate all permutations of parameters from IndexArgs."""
        base_params = product(self.ef_search, self.filter_selectivity)
        kwargs_combos = product(*self.kwargs.values()) if self.kwargs else [()]

        return [
            SearchArgs(
                ef_search=ef_search,
                filter_selectivity=filter_selectivity,
                kwargs=dict(zip(self.kwargs.keys(), kwarg_vals)),
            )
            for (ef_search, filter_selectivity), kwarg_vals in product(base_params, kwargs_combos)
        ]
