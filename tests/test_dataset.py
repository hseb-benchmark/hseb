from hseb.core.dataset import BenchmarkDataset
from hseb.core.config import DatasetConfig


class TestDataset:
    def test_dataset_loading(self):
        conf = DatasetConfig(
            dim=384,
            name="nixiesearch/benchmark-msmarco",
            query="query-all-MiniLM-L6-v2-1K",
            corpus="corpus-all-MiniLM-L6-v2-1K",
        )
        ds = BenchmarkDataset(conf)
        assert len(ds.corpus) == 1000
