from hseb.core.config import (
    Config,
    ExperimentConfig,
    DatasetConfig,
    IndexArgs,
    IndexArgsMatrix,
    SearchArgs,
    SearchArgsMatrix,
)
from hseb.engine.nixie.nixiesearch import Nixiesearch
from hseb.core.dataset import BenchmarkDataset
from tqdm import tqdm


class TestNixiesearch:
    def test_start_stop(self):
        config = Config(
            engine="nixiesearch",
            image="nixiesearch/nixiesearch:0.6.1-M1-amd64",
            dataset=DatasetConfig(
                dim=384,
                name="hseb-benchmark/msmarco",
                query="query-all-MiniLM-L6-v2-1K",
                corpus="corpus-all-MiniLM-L6-v2-1K",
            ),
            experiments=[
                ExperimentConfig(
                    tag="test",
                    index=IndexArgsMatrix(m=[16], ef_construction=[64], quant=["float32"]),
                    search=SearchArgsMatrix(ef_search=[16], filter_selectivity=[100]),
                )
            ],
        )

        data = BenchmarkDataset(config.dataset)
        server = Nixiesearch(config)
        for exp in config.experiments:
            for index_args in exp.index.expand():
                server.start(index_args)
                for batch in data.corpus_batches(index_args.batch_size):
                    server.index_batch(batch)

                server.commit()
                for search_args in exp.search.expand():
                    for query in tqdm(data.query, desc="searching"):
                        server.search(search_args, query, 10)
