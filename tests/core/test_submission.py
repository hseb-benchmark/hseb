import time
from hseb.core.config import (
    Config,
    DatasetConfig,
    ExperimentConfig,
    IndexArgs,
    IndexArgsMatrix,
    QuantDatatype,
    SearchArgs,
    SearchArgsMatrix,
)
from hseb.core.measurement import ExperimentResult, QueryResult
from hseb.core.response import DocScore
import tempfile
import os

from hseb.core.submission import ExperimentMetrics, Submission


def test_submission():
    config = Config(
        engine="hseb.engine.nixiesearch.nixiesearch.NixiesearchEngine",
        image="nixiesearch/nixiesearch:0.6.5",
        dataset=DatasetConfig(
            dim=384,
            name="hseb-benchmark/msmarco",
            query="query-all-MiniLM-L6-v2-1K",
            corpus="corpus-all-MiniLM-L6-v2-1K",
        ),
        experiments=[
            ExperimentConfig(
                tag="test",
                k=10,
                index=IndexArgsMatrix(m=[16], ef_construction=[64], quant=["float32"]),
                search=SearchArgsMatrix(ef_search=[16], filter_selectivity=[100]),
            )
        ],
    )
    m1 = QueryResult(query_id=1, exact=[DocScore(1, 1)], response=[DocScore(1, 1)], client_latency=1)
    m2 = QueryResult(query_id=1, exact=[DocScore(1, 1)], response=[DocScore(1, 1)], client_latency=1)
    result = ExperimentResult(
        tag="test",
        index_args=IndexArgs(m=32, ef_construction=32, quant=QuantDatatype.FLOAT32, batch_size=32, kwargs={}),
        search_args=SearchArgs(ef_search=32, filter_selectivity=100, kwargs={}),
        queries=[m1, m2],
        indexing_time=[1],
        warmup_latencies=[1],
    )
    with tempfile.TemporaryDirectory(prefix="hseb_test_") as dir:
        result.to_json(dir)
        sub = Submission.from_dir(config=config, path=dir)
        assert len(sub.experiments) == 1
        assert len(sub.experiments[0].metrics.latency) == 2
        export_path = os.path.join(tempfile.tempdir, f"hseb_export_test_{time.time()}.json")

        sub.to_json(export_path)
        decoded = Submission.from_json(export_path)
        assert decoded == sub


def test_recall():
    r1 = ExperimentMetrics.recall_score([1, 2, 3, 4], [1, 2, 4, 5], 4)
    assert r1 == 0.75
