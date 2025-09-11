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
            engine="hseb.engine.postgres.postgres.PostgresEngine",
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
                        ef_construction=[64],
                        quant=[QuantDatatype.FLOAT32],
                        kwargs={"shared_buffers": ["2GB"], "work_mem": ["16MB"], "maintenance_work_mem": ["512MB"]},
                    ),
                    search=SearchArgsMatrix(ef_search=[16], filter_selectivity=[100]),
                )
            ],
        )
