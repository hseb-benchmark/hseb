from abc import ABC, abstractmethod
import time
import math

from hseb.core.config import Config, DatasetConfig, ExperimentConfig, IndexArgsMatrix, QuantDatatype, SearchArgsMatrix
from hseb.core.dataset import BenchmarkDataset, Doc
from hseb.core.measurement import ExperimentResult, QueryResult
from hseb.core.submission import ExperimentMetrics
from hseb.engine.base import EngineBase
from tqdm import tqdm
from structlog import get_logger
from dataclasses import dataclass, field

logger = get_logger()


@dataclass
class EngineParams:
    engine: str
    image: str
    quantizations: list[QuantDatatype] = field(default_factory=list)
    index_kwargs: dict[str, list] = field(default_factory=dict)
    search_kwargs: dict[str, list] = field(default_factory=dict)
    order_tolerance_rate: float = 0.0


class EngineSuite(ABC):
    @abstractmethod
    def params(self) -> EngineParams: ...

    def config(self) -> Config:
        engine_params = self.params()
        return Config(
            engine=engine_params.engine,
            image=engine_params.image,
            dataset=DatasetConfig(
                dim=384,
                name="hseb-benchmark/msmarco",
                query="query-all-MiniLM-L6-v2-1K",
                corpus="corpus-all-MiniLM-L6-v2-1K",
            ),
            experiments=[
                ExperimentConfig(
                    tag="test",
                    k=10,
                    index=IndexArgsMatrix(
                        m=[16],
                        ef_construction=[64],
                        quant=[QuantDatatype.FLOAT32],
                        kwargs=engine_params.index_kwargs,
                    ),
                    search=SearchArgsMatrix(
                        ef_search=[16],
                        filter_selectivity=[10, 100],
                        kwargs=engine_params.search_kwargs,
                    ),
                ),
                ExperimentConfig(
                    tag="test-quant",
                    k=10,
                    index=IndexArgsMatrix(
                        m=[16],
                        ef_construction=[64],
                        quant=engine_params.quantizations,
                        kwargs=engine_params.index_kwargs,
                    ),
                    search=SearchArgsMatrix(ef_search=[16], filter_selectivity=[100]),
                ),
            ],
        )

    def test_index_search(self):
        conf: Config = self.config()
        data = BenchmarkDataset(conf.dataset)
        engine = EngineBase.load_class(conf.engine, config=conf)
        engine_params = self.params()

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

                            # Check ordering with tolerance
                            violations = 0
                            prev_score = 10000.0
                            for doc in results.results:
                                if doc.score > prev_score:
                                    violations += 1
                                assert isinstance(doc.doc, int)
                                real_doc = docs[doc.doc]
                                assert search_args.filter_selectivity in real_doc.tag
                                prev_score = doc.score

                            # Allow violations based on tolerance rate
                            max_violations = math.ceil(len(results.results) * engine_params.order_tolerance_rate)
                            assert violations <= max_violations, (
                                f"Too many ordering violations: {violations} > {max_violations}"
                            )
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
