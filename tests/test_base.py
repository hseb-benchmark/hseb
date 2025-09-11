from hseb.core.config import Config
from hseb.engine.base import EngineBase


def test_load_class():
    config = Config.from_file("configs/nixiesearch/dev.yml")
    engine = EngineBase.load_class(name="hseb.engine.nixiesearch.NixiesearchEngine", config=config)
    assert isinstance(engine, EngineBase)
