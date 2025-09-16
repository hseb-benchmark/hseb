from hseb.engine.base import EngineBase
import requests
import json
import time
from hseb.core.response import DocScore, IndexResponse, SearchResponse
import tempfile
import yaml
from hseb.core.config import Config, SearchArgs, IndexArgs
from hseb.core.dataset import Doc, Query
import docker
import logging

logger = logging.getLogger()


class NixiesearchEngine(EngineBase):
    def __init__(self, config: Config):
        self.config = config

    def index_batch(self, batch: list[Doc]) -> IndexResponse:
        start = time.perf_counter()
        payload = []
        for doc in batch:
            doc_json = {
                "_id": doc.id,
                "text": {"text": doc.text, "embedding": doc.embedding.tolist()},
                "tag": doc.tag,
            }
            payload.append(doc_json)
        start = time.perf_counter()
        response = requests.post("http://localhost:8080/v1/index/test", json=payload)
        end = time.perf_counter()
        logger.debug(f"Indexed batch of {len(batch)} docs in {time.perf_counter() - start} sec")
        if response.status_code != 200:
            raise Exception(response.text)
        if "refresh_every" in self.index_args.kwargs:
            self.docs_in_segment += len(batch)
            if self.docs_in_segment >= self.index_args.kwargs["refresh_every"]:
                requests.post("http://localhost:8080/v1/index/test/flush")
                self.docs_in_segment = 0
        return IndexResponse(client_latency=end - start)

    def commit(self):
        logger.debug(requests.post("http://localhost:8080/v1/index/test/flush"))
        if self.index_args.segments is not None:
            requests.post("http://localhost:8080/v1/index/test/merge", json={"segments": self.index_args.segments})

    def search(self, search_params: SearchArgs, query: Query, top_k: int) -> SearchResponse:
        payload = {
            "query": {
                "knn": {
                    "field": "text",
                    "query_vector": query.embedding.tolist(),
                    "num_candidates": search_params.ef_search,
                    "k": top_k,
                }
            },
            "size": top_k,
        }
        if search_params.filter_selectivity != 100:
            payload["filters"] = {"include": {"term": {"tag": search_params.filter_selectivity}}}
        start = time.time_ns()

        response = requests.post("http://localhost:8080/v1/index/test/search", json=payload)
        if response.status_code != 200:
            logger.error(payload)
            raise Exception(f"got non-200 response: {response.text}")
        end = time.time_ns()
        decoded = json.loads(response.text)
        results = [DocScore(doc=int(hit["_id"]), score=float(hit["_score"])) for hit in decoded["hits"]]
        return SearchResponse(results=results, client_latency=(end - start) / 1000000000.0)

    def start(self, index_args: IndexArgs):
        self.index_args = index_args
        indexer_config = {"ram_buffer_size": index_args.kwargs.get("ram_buffer_size", "512mb")}
        if "refresh_every" in index_args.kwargs:
            indexer_config["flush"] = {"interval": "1h"}  # manual control
        if "max_merge_docs" in index_args.kwargs:
            indexer_config["merge_policy"] = {"doc_count": {"max_merge_docs": index_args.kwargs["max_merge_docs"]}}
        engine_config_file = {
            "schema": {
                "test": {
                    "config": {"indexer": indexer_config},
                    "fields": {
                        "text": {
                            "type": "text",
                            "search": {
                                "semantic": {
                                    "m": index_args.m,
                                    "ef": index_args.ef_construction,
                                    "dim": self.config.dataset.dim,
                                    "quantize": index_args.quant.value,
                                }
                            },
                        },
                        "tag": {"type": "int[]", "filter": True},
                    },
                }
            },
        }

        self.dir = tempfile.TemporaryDirectory()
        # self.container = None
        with open(f"{self.dir.name}/config.yml", "w") as config_file:
            config_file.write(yaml.safe_dump(engine_config_file))
        docker_client = docker.from_env()
        heap_size = index_args.kwargs.get("heap_size", "8g")
        self.container = docker_client.containers.run(
            image=self.config.image,
            ports={"8080/tcp": 8080},
            volumes={self.dir.name: {"bind": "/data", "mode": "rw"}},
            command="standalone -c /data/config.yml --loglevel debug",
            environment={
                "JAVA_OPTS": f"-Xmx{heap_size} -Xms{heap_size} -verbose:gc --add-modules jdk.incubator.vector"
            },
            detach=True,
        )
        self._wait_for_logs(self.container, "Ember-Server service bound to address")
        self.docs_in_segment = 0
        return self

    def stop(self, cleanup: bool):
        self.container.stop()
        self.dir.cleanup()
        if cleanup:
            self.container.remove()
