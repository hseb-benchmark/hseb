from hseb.core.dataset import BenchmarkDataset
from hseb.core.config import DatasetConfig


class TestDataset:
    def test_dataset_loading(self):
        conf = DatasetConfig(
            dim=384,
            name="hseb-benchmark/msmarco",
            query="query-all-MiniLM-L6-v2-1K",
            corpus="corpus-all-MiniLM-L6-v2-1K",
        )
        ds = BenchmarkDataset(conf)
        assert len(ds.corpus_dataset) == 1000
        assert len(list(ds.corpus_batched(100))) == 10
        assert len(ds.query_dataset) == 10000
        assert len(list(ds.queries())) == 10000
