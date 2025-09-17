from hseb.core.config import (
    Config,
    ExperimentConfig,
    DatasetConfig,
    IndexArgsMatrix,
    QuantDatatype,
    SearchArgsMatrix,
)

from tests.engine.base import EngineSuite


class TestRedisEngine(EngineSuite):
    def config(self) -> Config:
        return Config(
            engine="hseb.engine.redis.RedisEngine",
            image="redis:8.2.1",
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
                    index=IndexArgsMatrix(
                        m=[16],
                        ef_construction=[32],
                        quant=[QuantDatatype.FLOAT32],
                        kwargs={"maxmemory": ["2gb"], "maxmemory_policy": ["allkeys-lru"]},
                    ),
                    search=SearchArgsMatrix(ef_search=[128], filter_selectivity=[10, 100]),
                )
            ],
        )
