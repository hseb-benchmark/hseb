from hseb.core.config import (
    QuantDatatype,
)

from tests.engine.base import EngineParams, EngineSuite


class TestElasticengine(EngineSuite):
    def params(self):
        return EngineParams(
            engine="hseb.engine.elastic.ElasticsearchEngine",
            image="elasticsearch:9.1.3",
            quantizations=[QuantDatatype.INT8, QuantDatatype.INT1],
        )
