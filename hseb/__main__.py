import argparse
import time
from hseb.engine.base import EngineBase
from hseb.core.config import Config
from hseb.core.dataset import BenchmarkDataset
from hseb.core.measurement import ExperimentResult, Measurement
from tqdm import tqdm
import json
from dataclasses import asdict
from structlog import get_logger

logger = get_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--outdir", type=str, required=True)

    args = parser.parse_args()

    config = Config.from_file(args.config)
    data = BenchmarkDataset(config.dataset)
    engine = EngineBase.load_class(config.engine, config)
    logger.info("Initialized engine")
    run_index = 0
    for exp_index, exp in enumerate(config.experiments):
        index_variations = exp.index.expand()
        search_variations = exp.search.expand()
        total_cases = len(index_variations) * len(search_variations)
        logger.info(
            f"Running experiment {exp_index} of {len(config.experiments)}. Targets: index={len(index_variations)} search={len(search_variations)} total={total_cases}"
        )

        for indexing_args_index, index_args in enumerate(index_variations):
            logger.info(f"Indexing run {indexing_args_index}/{len(index_variations)}: {index_args}")
            try:
                index_start = time.perf_counter()
                engine.start(index_args)
                batches = data.corpus_batched(index_args.batch_size)
                total = len(data.corpus_dataset) / index_args.batch_size
                for batch in tqdm(batches, total=total, desc="indexing"):
                    engine.index_batch(batch)
                engine.commit()
                warmup_start = time.perf_counter()
                for warmup_query in tqdm(list(data.queries()), desc="warmup"):
                    response = engine.search(search_variations[0], warmup_query, 100)
                logger.info(f"Warmup done in {time.perf_counter() - warmup_start} seconds")
                for search_args_index, search_args in enumerate(search_variations):
                    logger.info(
                        f"Search {search_args_index}/{len(search_variations)} ({run_index}/{total_cases}): {search_args}"
                    )
                    out_file = f"{args.outdir}/{exp.tag}-{index_args.to_string()}-{search_args.to_string()}.json"
                    with open(out_file, "w") as out:
                        measurements: list[Measurement] = []

                        for query in tqdm(list(data.queries()), desc="search"):
                            response = engine.search(search_args, query, 100)
                            measurements.append(
                                Measurement.from_response(query_id=query.id, exact=query.exact100, response=response)
                            )

                        result = ExperimentResult(
                            tag=exp.tag,
                            index_args=index_args,
                            search_args=search_args,
                            measurements=measurements,
                        )

                        out.write(json.dumps(asdict(result)))

                    run_index += 1
                logger.debug(
                    f"Indexing run {indexing_args_index}/{len(index_variations)} done in {time.perf_counter() - index_start} seconds"
                )
            finally:
                engine.stop()
    logger.info("Benchmark finished.")
