from abc import ABC, abstractmethod

from hseb.core.config import Config
from hseb.core.dataset import BenchmarkDataset
from hseb.engine.base import EngineBase
from tqdm import tqdm


class EngineSuite(ABC):
    @abstractmethod
    def config(self) -> Config: ...

    def test_start_stop(self):
        conf: Config = self.config()
        engine = EngineBase.load_class(conf.engine, config=conf)
        for exp in conf.experiments:
            for index_args in exp.index.expand():
                try:
                    engine.start(index_args)
                finally:
                    engine.stop(cleanup=True)

    def test_index_search(self):
        conf: Config = self.config()
        data = BenchmarkDataset(conf.dataset)
        engine = EngineBase.load_class(conf.engine, config=conf)

        for exp in conf.experiments:
            for index_args in exp.index.expand():
                try:
                    engine.start(index_args)
                    for batch in data.corpus_batched(index_args.batch_size):
                        engine.index_batch(batch)

                    engine.commit()
                    for search_args in exp.search.expand():
                        for query in tqdm(list(data.queries())[:10], desc="searching"):
                            results = engine.search(search_args, query, 16)
                            assert len(results.results) == 16
                            prev_score = 10000.0
                            for doc in results.results:
                                assert doc.score > 0.0
                                assert doc.score <= prev_score
                                assert isinstance(doc.doc, int)
                                prev_score = doc.score
                finally:
                    engine.stop(cleanup=True)
