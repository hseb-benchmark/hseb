from hseb.core.config import (
    QuantDatatype,
)

from tests.engine.base import EngineParams, EngineSuite


class TestOpensearch3Engine(EngineSuite):
    def params(self):
        return EngineParams(
            engine="hseb.engine.opensearch.OpenSearchEngine",
            image="opensearchproject/opensearch:3.2.0",
            quantizations=[QuantDatatype.INT8, QuantDatatype.FLOAT16],
            index_kwargs={"engine": ["lucene", "faiss"]},
        )
