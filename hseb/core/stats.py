from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PERCENTILES = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]


def recall(approx: list[int], exact: list[int], k: int) -> float:
    count = min(len(approx), len(exact), k)
    return len(set(approx[:count]).intersection(exact[:count])) / float(count)


@dataclass
class Stats:
    percentiles: dict[str, float]
    avg: float

    @staticmethod
    def from_list(values: list[float]) -> Stats:
        perc = list(np.percentile(values, PERCENTILES))
        avg = np.mean(values)
        return Stats(percentiles={str(p): v for p, v in zip(PERCENTILES, perc)}, avg=float(avg))
