from hseb.core.config import (
    QuantDatatype,
)

from tests.engine.base import EngineParams, EngineSuite


class TestOpensearch2Engine(EngineSuite):
    def params(self):
        return EngineParams(
            engine="hseb.engine.opensearch.OpenSearchEngine",
            image="opensearchproject/opensearch:2.19.2",
            quantizations=[QuantDatatype.INT8, QuantDatatype.FLOAT16],
            index_kwargs={"engine": ["lucene", "faiss"]},
        )
