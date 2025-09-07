from __future__ import annotations
import yaml
from typing import Any
from itertools import product
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict
from abc import ABC


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
            return Config.from_yaml(f.read())


class DatasetConfig(BaseModel):
    dim: int = Field(gt=0)
    name: str
    query: str = Field(description="queries dataset namr")
    corpus: str = Field(description="corpus dataset name")


class ExperimentConfig(BaseModel):
    tag: str
    index: IndexArgsMatrix
    search: SearchArgsMatrix


@dataclass
class IndexArgs:
    m: int
    ef_construction: int
    quant: str
    batch_size: int
    kwargs: dict[str, Any]

    def to_string(self) -> str:
        parts1 = [f"{key}={value}" for key, value in asdict(self).items() if key != "kwargs"]
        parts2 = [f"{key}={value}" for key, value in self.kwargs.items()]
        return "_".join(parts1 + parts2)


class IndexArgsMatrix(BaseModel):
    m: list[int]
    ef_construction: list[int]
    quant: list[str]
    batch_size: int = 1024
    kwargs: dict[str, list] = Field(default_factory=dict)

    @staticmethod
    def from_dict(input: dict) -> IndexArgsMatrix:
        defined_fields = {"m", "ef_construction", "quant", "batch_size"}
        regular_fields = {k: v for k, v in input.items() if k in defined_fields}
        extra_fields = {k: v for k, v in input.items() if k not in defined_fields}

        return IndexArgsMatrix(**regular_fields, kwargs=extra_fields)

    def expand(self) -> list[IndexArgs]:
        """Generate all permutations of parameters from IndexArgs."""
        base_params = product(self.m, self.ef_construction, self.quant)
        kwargs_combos = product(*self.kwargs.values()) if self.kwargs else [()]

        return [
            IndexArgs(
                m=m,
                ef_construction=ef,
                quant=quant,
                batch_size=self.batch_size,
                kwargs=dict(zip(self.kwargs.keys(), kwarg_vals)),
            )
            for (m, ef, quant), kwarg_vals in product(base_params, kwargs_combos)
        ]


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

    @staticmethod
    def from_dict(input: dict) -> SearchArgsMatrix:
        defined_fields = {"ef_search", "filter_selectivity"}
        regular_fields = {k: v for k, v in input.items() if k in defined_fields}
        extra_fields = {k: v for k, v in input.items() if k not in defined_fields}
        return SearchArgsMatrix(**regular_fields, kwargs=extra_fields)

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
