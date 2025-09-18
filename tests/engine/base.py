from abc import ABC, abstractmethod
import time

from hseb.core.config import Config
from hseb.core.dataset import BenchmarkDataset, Doc
from hseb.core.measurement import ExperimentResult, QueryResult
from hseb.core.submission import ExperimentMetrics
from hseb.engine.base import EngineBase
from tqdm import tqdm
from structlog import get_logger

logger = get_logger()


class EngineSuite(ABC):
    @abstractmethod
    def config(self) -> Config: ...

    def test_index_search(self):
        conf: Config = self.config()
        data = BenchmarkDataset(conf.dataset)
        engine = EngineBase.load_class(conf.engine, config=conf)

        docs: dict[int, Doc] = {doc.id: doc for doc in data.corpus()}

        for exp in conf.experiments:
            for index_args in exp.index.expand():
                try:
                    logger.info("starting engine")
                    engine.start(index_args)
                    logger.info(f"indexing: {index_args}")
                    for batch in data.corpus_batched(index_args.batch_size):
                        engine.index_batch(batch)

                    engine.commit()
                    is_green = False
                    attempts = 0
                    while not is_green and attempts < 30:
                        is_green = engine.index_is_green()
                        time.sleep(1)
                    assert is_green
                    for search_args in exp.search.expand():
                        measurements = []
                        logger.info(f"searching: {search_args}")
                        for query in tqdm(list(data.queries(limit=10)), desc="searching"):
                            results = engine.search(search_args, query, 16)
                            assert len(results.results) == 16
                            measurements.append(QueryResult.from_response(query, search_args, results))
                            prev_score = 10000.0
                            for doc in results.results:
                                assert doc.score <= prev_score
                                assert isinstance(doc.doc, int)
                                real_doc = docs[doc.doc]
                                assert search_args.filter_selectivity in real_doc.tag
                                prev_score = doc.score
                        result = ExperimentResult(
                            tag="test",
                            indexing_time=[1],
                            index_args=index_args,
                            search_args=search_args,
                            queries=measurements,
                            warmup_latencies=[1],
                        )
                        metrics = ExperimentMetrics.from_experiment(result)
                        recall10 = sum(metrics.metrics.recall10) / len(metrics.metrics.recall10)
                        assert recall10 > 0.01
                finally:
                    engine.stop(cleanup=True)
