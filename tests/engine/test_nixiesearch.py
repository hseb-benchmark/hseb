from hseb.core.config import (
    QuantDatatype,
)

from tests.engine.base import EngineParams, EngineSuite


class TestNixiesearch(EngineSuite):
    def params(self):
        return EngineParams(
            engine="hseb.engine.nixiesearch.NixiesearchEngine",
            image="nixiesearch/nixiesearch:0.7.2",
            quantizations=[QuantDatatype.INT8, QuantDatatype.INT1],
        )
