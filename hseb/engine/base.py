from __future__ import annotations
from typing import Type
from hseb.core.response import IndexResponse, SearchResponse
import time
from abc import ABC, abstractmethod
from hseb.core.dataset import Query, Doc
from hseb.core.config import Config, SearchArgs, IndexArgs
import logging
import importlib

logger = logging.getLogger()


class EngineBase(ABC):
    @abstractmethod
    def start(self, index_args: IndexArgs) -> None: ...

    @abstractmethod
    def stop(self, cleanup: bool) -> None: ...

    @abstractmethod
    def commit(self) -> None: ...

    @abstractmethod
    def index_is_green(self) -> bool: ...

    @abstractmethod
    def index_batch(self, batch: list[Doc]) -> IndexResponse: ...

    def pre_search(self, index_args: IndexArgs, search_args: SearchArgs) -> None:
        # do nothing by default
        return None

    @abstractmethod
    def search(self, search_args: SearchArgs, query: Query, top_k: int) -> SearchResponse: ...

    def _wait_for_logs(self, container, log_message: str, timeout: int = 60):
        """Wait for a specific log message to appear in container logs"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            logs = container.logs().decode("utf-8")
            # print(logs)
            if log_message in logs:
                return True
            time.sleep(1)
        raise TimeoutError(f"Timeout waiting for log message: {log_message}")

    @staticmethod
    def load_class(name: str, config: Config) -> EngineBase:
        module_name, class_name = name.rsplit(".", 1)
        logger.debug(f"Loading model {class_name} from module {module_name}")
        module = importlib.import_module(module_name)
        if not hasattr(module, class_name):
            raise ValueError(f"Cannot find {class_name} in module {module_name}")
        cls: Type[EngineBase] = getattr(module, class_name)
        if not issubclass(cls, EngineBase):
            raise ValueError(f"Class {class_name} is not a EngineBase")

        obj = cls(config=config)
        return obj
