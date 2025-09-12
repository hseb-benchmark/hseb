import argparse
import time
from hseb.engine.base import EngineBase
from hseb.core.config import Config
from hseb.core.dataset import BenchmarkDataset
from hseb.core.measurement import ExperimentResult, Measurement, Submission
from tqdm import tqdm
from structlog import get_logger
import tempfile

logger = get_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="path to a config file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="output file name",
    )
    parser.add_argument(
        "--delete-container",
        type=bool,
        required=False,
        default=False,
        help="should we delete all stopped containers? this saves disk space, but you lose engine logs",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config)
    data = BenchmarkDataset(config.dataset)
    engine = EngineBase.load_class(config.engine, config)
    with tempfile.TemporaryDirectory(prefix="hseb_", delete=False) as workdir:
        logger.info(f"Initialized engine, workdir: {workdir}")
        run_index = 0
        for exp_index, exp in enumerate(config.experiments):
            index_fails = 0
            search_fails = 0
            index_variations = exp.index.expand()
            search_variations = exp.search.expand()
            total_cases = len(index_variations) * len(search_variations)
            logger.info(
                f"Running experiment {exp_index + 1} of {len(config.experiments)}. Targets: index={len(index_variations)} search={len(search_variations)} total={total_cases}"
            )

            for indexing_args_index, index_args in enumerate(index_variations):
                logger.info(f"Indexing run {indexing_args_index + 1}/{len(index_variations)}: {index_args}")
                try:
                    engine.start(index_args)
                    batches = data.corpus_batched(index_args.batch_size)
                    total = int(len(data.corpus_dataset) / index_args.batch_size)
                    index_start = time.perf_counter()
                    for batch in tqdm(batches, total=total, desc="indexing"):
                        engine.index_batch(batch=batch)
                    commit_start = time.perf_counter()
                    engine.commit()
                    try:
                        warmup_start = time.perf_counter()
                        logger.info(
                            f"Index built in {warmup_start - index_start} seconds (ingest={int(commit_start - index_start)} commit={int(warmup_start - commit_start)})"
                        )
                        warmup_latencies: list[float] = []
                        for warmup_query in tqdm(list(data.queries()), desc="warmup"):
                            response = engine.search(search_variations[0], warmup_query, exp.k)
                            warmup_latencies.append(response.client_latency)

                        logger.info(f"Warmup done in {time.perf_counter() - warmup_start} seconds")
                        for search_args_index, search_args in enumerate(search_variations):
                            logger.info(
                                f"Search {search_args_index + 1}/{len(search_variations)} ({run_index + 1}/{total_cases}): {search_args}"
                            )

                            measurements: list[Measurement] = []

                            for query in tqdm(list(data.queries()), desc="search"):
                                response = engine.search(search_args, query, exp.k)
                                if len(response.results) != exp.k:
                                    logger.warn(
                                        f"Engine returned {len(response.results)} docs, which less than {exp.k} docs expected"
                                    )
                                measurements.append(
                                    Measurement.from_response(
                                        query_id=query.id, exact=query.exact100, response=response
                                    )
                                )

                            result = ExperimentResult(
                                tag=exp.tag,
                                index_args=index_args,
                                search_args=search_args,
                                measurements=measurements,
                                indexing_time=warmup_start - index_start,
                                warmup_latencies=warmup_latencies,
                            )
                            result.to_json(workdir=workdir)

                            run_index += 1
                    except Exception:
                        logger.error(f"skipping search run {search_args}")
                        search_fails += 1
                    logger.debug(
                        f"Indexing run {indexing_args_index + 1}/{len(index_variations)} done in {time.perf_counter() - index_start} seconds"
                    )
                except Exception:
                    logger.error(f"skipping index run {index_args}")
                    index_fails += 1
                finally:
                    engine.stop()
        submission = Submission.from_dir(config=config, path=workdir)
        logger.info(f"Writing submission file to {args.out}")
        submission.to_json(args.out)
        logger.info(f"Experiment finished. {index_fails} index fails, {search_fails} search fails.")
