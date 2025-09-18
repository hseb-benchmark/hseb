from hseb.core.config import (
    QuantDatatype,
)

from tests.engine.base import EngineParams, EngineSuite


class TestQdrantEngine(EngineSuite):
    def params(self):
        return EngineParams(
            engine="hseb.engine.qdrant.Qdrant",
            image="qdrant/qdrant:v1.15.4",
            quantizations=[QuantDatatype.FLOAT16, QuantDatatype.INT8, QuantDatatype.INT1],
        )
