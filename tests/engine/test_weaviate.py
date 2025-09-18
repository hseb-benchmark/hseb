from hseb.core.config import (
    QuantDatatype,
)

from tests.engine.base import EngineParams, EngineSuite


class TestWeaviateEngine(EngineSuite):
    def params(self):
        return EngineParams(
            engine="hseb.engine.weaviate.WeaviateEngine",
            image="semitechnologies/weaviate:1.32.8",
            quantizations=[QuantDatatype.INT8, QuantDatatype.INT1],
        )
