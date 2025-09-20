from hseb.core.config import IndexArgs, QuantDatatype, SearchArgs
from hseb.core.measurement import ExperimentResult, QueryResult
from hseb.core.report import Report
from hseb.core.response import DocScore
from hseb.core.submission import ExperimentMetrics, Submission


def test_report():
    m = QueryResult(query_id=1, exact=[DocScore(1, 1)], response=[DocScore(1, 1)], client_latency=1)
    result = ExperimentResult(
        tag="test",
        index_args=IndexArgs(
            m=32,
            ef_construction=32,
            quant=QuantDatatype.FLOAT32,
            batch_size=32,
            segments=None,
            kwargs={},
        ),
        search_args=SearchArgs(ef_search=32, filter_selectivity=100, kwargs={}),
        queries=[m],
        indexing_time=[1],
        warmup_latencies=[1],
    )
    submission = Report.from_experiments([ExperimentMetrics.from_experiment(result)])
    assert submission.df.shape == (1, 18)
