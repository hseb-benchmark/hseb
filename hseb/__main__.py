import argparse
from hseb.engine.nixiesearch.nixiesearch import Nixiesearch
from hseb.core.config import Config, ExperimentConfig
from hseb.core.dataset import BenchmarkDataset
from hseb.core.measurement import Measurement, ExperimentResult
from tqdm import tqdm
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--outdir", type=str, required=True)

    args = parser.parse_args()

    config = Config.from_file(args.config)
    data = BenchmarkDataset(config.dataset)

    match config.engine:
        case "nixiesearch":
            for exp in config.experiments:
                index_variations = exp.index.expand()
                search_variations = exp.search.expand()
                total_cases = len(index_variations) * len(search_variations) * (len(data.query_dataset[:1000]) + 100)
                with tqdm(total=total_cases, desc="progress") as progress_bar:
                    for index_args in index_variations:
                        engine = Nixiesearch(config)
                        engine.start(index_args)
                        engine.index(data.corpus_dataset)
                        for search_params in search_variations:
                            out_file = (
                                f"{args.outdir}/{exp.tag}-{index_params.to_string()}-{search_params.to_string()}.json"
                            )
                            with open(out_file, "w") as out:
                                result = Result(
                                    tag=exp.tag,
                                    index_params=index_params,
                                    search_params=search_params,
                                    measurements=[],
                                )
                                for warmup_query in data.query_dataset[:100]:
                                    response = engine.search(search_params, warmup_query, 100)
                                    progress_bar.update()
                                for query in data.query_dataset[:1000]:
                                    response = engine.search(search_params, query, 100)
                                    meas = Measurement.from_response(query, response)
                                    result.measurements.append(meas)
                                    progress_bar.update()
                                out.write(json.dumps(result.to_dict()))
