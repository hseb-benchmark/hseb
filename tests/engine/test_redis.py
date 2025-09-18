from hseb.core.config import (
    QuantDatatype,
)

from tests.engine.base import EngineParams, EngineSuite


class TestRedisEngine(EngineSuite):
    def params(self):
        return EngineParams(
            engine="hseb.engine.redis.RedisEngine",
            image="redis:8.2.1",
            quantizations=[QuantDatatype.FLOAT16],
        )
