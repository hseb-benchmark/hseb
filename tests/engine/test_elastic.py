from hseb.core.config import (
    Config,
    ExperimentConfig,
    DatasetConfig,
    IndexArgsMatrix,
    QuantDatatype,
    SearchArgsMatrix,
)

from tests.engine.base import EngineSuite


class TestElasticengine(EngineSuite):
    def config(self) -> Config:
        return Config(
            engine="hseb.engine.elastic.ElasticsearchEngine",
            image="elasticsearch:9.1.3",
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
                    index=IndexArgsMatrix(m=[16], ef_construction=[64], quant=[QuantDatatype.FLOAT32]),
                    search=SearchArgsMatrix(ef_search=[16], filter_selectivity=[100]),
                )
            ],
        )
