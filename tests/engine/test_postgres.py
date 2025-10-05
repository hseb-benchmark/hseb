from hseb.core.config import (
    QuantDatatype,
)

from tests.engine.base import EngineParams, EngineSuite


class TestPostgresEngine(EngineSuite):
    def params(self):
        return EngineParams(
            engine="hseb.engine.postgres.PostgresEngine",
            image="pgvector/pgvector:0.8.1-pg17-trixie",
            quantizations=[QuantDatatype.FLOAT16],
            order_tolerance_rate=0.05,
            search_kwargs={"order": ["relaxed_order", "strict_order"]},
        )
