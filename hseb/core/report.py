from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from hseb.core.submission import ExperimentMetrics
from tqdm import tqdm


@dataclass
class Report:
    df: pd.DataFrame

    @staticmethod
    def from_experiments(engine: str, experiments: list[ExperimentMetrics]) -> Report:
        columns: dict[str, list] = defaultdict(list)
        for exp in tqdm(experiments, desc="processing experiments"):
            columns["db"].append(engine)
            columns["tag"].append(exp.tag)
            columns["indexing_time_total"].append(sum(exp.indexing_time))
            columns["indexing_time_batch_mean"].append(sum(exp.indexing_time) / len(exp.indexing_time))
            columns["indexing_time_batch_median"].append(np.percentile(exp.indexing_time, 50).item())
            columns["ef_construction"].append(exp.index_args.ef_construction)
            columns["m"].append(exp.index_args.m)
            columns["quant"].append(exp.index_args.quant)
            columns["batch_size"].append(exp.index_args.batch_size)
            columns["segments"].append(exp.index_args.segments)
            for key, value in exp.index_args.kwargs.items():
                columns[key].append(value)
            columns["index_args"].append(exp.index_args.to_string())
            columns["ef_search"].append(exp.search_args.ef_search)
            columns["filter_selectivity"].append(exp.search_args.filter_selectivity)
            for key, value in exp.search_args.kwargs.items():
                columns[key].append(value)
            for metric, value in exp.metrics.as_dict().items():
                columns[metric].append(value)
        for col, values in columns.items():
            print(f"{col}={len(values)}")
        return Report(pd.DataFrame(columns))
