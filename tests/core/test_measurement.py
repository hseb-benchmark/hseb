from hseb.core.config import IndexArgs, QuantDatatype, SearchArgs
from hseb.core.measurement import ExperimentResult, Measurement
from hseb.core.response import DocScore
import tempfile
from pathlib import Path


def test_write_read():
    m = Measurement(query_id=1, exact=[DocScore(1, 1)], response=[DocScore(1, 1)], client_latency=1)
    result = ExperimentResult(
        tag="test",
        index_args=IndexArgs(m=32, ef_construction=32, quant=QuantDatatype.FLOAT32, batch_size=32, kwargs={}),
        search_args=SearchArgs(ef_search=32, filter_selectivity=100, kwargs={}),
        measurements=[m],
        indexing_time=1
    )
    with tempfile.TemporaryDirectory(prefix="hseb_test_") as dir:
        result.to_json(dir)
        for file in Path(dir).iterdir():
            decoded = ExperimentResult.from_json(file)
            assert decoded == result
