from hseb.core.config import (
    Config,
    ExperimentConfig,
    DatasetConfig,
    IndexArgsMatrix,
    QuantDatatype,
    SearchArgsMatrix,
)

from tests.engine.base import EngineSuite


class TestPostgresEngine(EngineSuite):
    def config(self) -> Config:
        return Config(
            engine="hseb.engine.postgres.PostgresEngine",
            image="pgvector/pgvector:0.8.1-pg17-trixie",
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
                    ),
                    search=SearchArgsMatrix(ef_search=[16], filter_selectivity=[10, 100]),
                )
            ],
        )
