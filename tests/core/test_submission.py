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
from hseb.core.measurement import ExperimentResult, Measurement, Submission
from hseb.core.response import DocScore
import tempfile


def test_submission():
    config = Config(
        engine="hseb.engine.nixiesearch.nixiesearch.NixiesearchEngine",
        image="nixiesearch/nixiesearch:0.6.3",
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
    m = Measurement(query_id=1, exact=[DocScore(1, 1)], response=[DocScore(1, 1)], client_latency=1)
    result = ExperimentResult(
        tag="test",
        index_args=IndexArgs(m=32, ef_construction=32, quant=QuantDatatype.FLOAT32, batch_size=32, kwargs={}),
        search_args=SearchArgs(ef_search=32, filter_selectivity=100, kwargs={}),
        measurements=[m, m, m],
    )
    with tempfile.TemporaryDirectory(prefix="hseb_test_") as dir:
        result.to_json(dir)
        sub = Submission.from_dir(config=config, path=dir)
        with tempfile.NamedTemporaryFile(prefix="hseb_test_result_") as tmp_file:
            sub.to_json(str(tmp_file.name))
            decoded = Submission.from_json(tmp_file.name)
            assert decoded == sub
