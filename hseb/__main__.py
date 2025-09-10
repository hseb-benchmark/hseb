import argparse
import time
from hseb.engine.base import EngineBase
from hseb.core.config import Config
from hseb.core.dataset import BenchmarkDataset
from hseb.core.measurement import ExperimentResult, Measurement, Submission
from tqdm import tqdm
import json
from dataclasses import asdict
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
    with tempfile.TemporaryDirectory(prefix="hseb", delete=False) as workdir:
        logger.info(f"Initialized engine, workdir: {workdir}")
        run_index = 0
        for exp_index, exp in enumerate(config.experiments):
            index_variations = exp.index.expand()
            search_variations = exp.search.expand()
            total_cases = len(index_variations) * len(search_variations)
            logger.info(
                f"Running experiment {exp_index + 1} of {len(config.experiments)}. Targets: index={len(index_variations)} search={len(search_variations)} total={total_cases}"
            )

            for indexing_args_index, index_args in enumerate(index_variations):
                logger.info(f"Indexing run {indexing_args_index + 1}/{len(index_variations)}: {index_args}")
                try:
                    index_start = time.perf_counter()
                    engine.start(index_args)
                    batches = data.corpus_batched(index_args.batch_size)
                    total = int(len(data.corpus_dataset) / index_args.batch_size)
                    for batch in tqdm(batches, total=total, desc="indexing"):
                        engine.index_batch(batch=batch, index_args=index_args)
                    commit_start = time.perf_counter()
                    engine.commit()
                    warmup_start = time.perf_counter()
                    logger.info(
                        f"Index built in {warmup_start - index_start} seconds (ingest={int(commit_start - index_start)} commit={int(warmup_start - commit_start)})"
                    )

                    for warmup_query in tqdm(list(data.queries()), desc="warmup"):
                        response = engine.search(search_variations[0], warmup_query, exp.k)
                    logger.info(f"Warmup done in {time.perf_counter() - warmup_start} seconds")
                    for search_args_index, search_args in enumerate(search_variations):
                        logger.info(
                            f"Search {search_args_index + 1}/{len(search_variations)} ({run_index + 1}/{total_cases}): {search_args}"
                        )
                        out_file = f"{workdir}/{exp.tag}-{index_args.to_string()}-{search_args.to_string()}.json"

                        measurements: list[Measurement] = []

                        for query in tqdm(list(data.queries()), desc="search"):
                            response = engine.search(search_args, query, exp.k)
                            if len(response.results) != exp.k:
                                logger.warn(
                                    f"Engine returned {len(response.results)} docs, which less than {exp.k} docs expected"
                                )
                            measurements.append(
                                Measurement.from_response(query_id=query.id, exact=query.exact100, response=response)
                            )

                        result = ExperimentResult(
                            tag=exp.tag,
                            index_args=index_args,
                            search_args=search_args,
                            measurements=measurements,
                        )
                        result.to_json(out_file)

                        run_index += 1
                    logger.debug(
                        f"Indexing run {indexing_args_index + 1}/{len(index_variations)} done in {time.perf_counter() - index_start} seconds"
                    )
                finally:
                    engine.stop()
        submission = Submission.from_dir(config=config, path=workdir)
        logger.info(f"Writing submission file to {args.out}")
        submission.to_json(args.out)
        logger.info("Benchmark finished.")
