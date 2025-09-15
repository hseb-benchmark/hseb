import logging
import time

import docker
from elasticsearch import Elasticsearch, helpers

from hseb.core.config import Config, IndexArgs, QuantDatatype, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import DocScore, IndexResponse, SearchResponse
from hseb.engine.base import EngineBase

logger = logging.getLogger()

ES_DATATYPES = {
    QuantDatatype.FLOAT32: "float",
    QuantDatatype.INT8: "byte",
    QuantDatatype.INT1: "bit",
}

ES_HNSW_TYPES = {
    QuantDatatype.FLOAT32: "hnsw",
    QuantDatatype.INT8: "int8_hnsw",
    QuantDatatype.INT1: "bbq_hnsw",
}


class ElasticsearchEngine(EngineBase):
    def __init__(self, config: Config):
        self.config = config

    def start(self, index_args: IndexArgs):
        self.index_args = index_args
        docker_client = docker.from_env()
        heap_size = index_args.kwargs.get("heap_size", "8g")
        self.container = docker_client.containers.run(
            image=self.config.image,
            ports={"9200/tcp": 9200, "9300/tcp": 9300},
            detach=True,
            environment={
                "discovery.type": "single-node",
                "ES_JAVA_OPTS": f"-Xms{heap_size} -Xmx{heap_size}",
                "xpack.security.enabled": "false",
            },
        )
        self._wait_for_logs(self.container, "is selected as the current health node")
        self.client = Elasticsearch("http://localhost:9200", request_timeout=30)
        # we control segment size by calling refresh
        settings = {"index": {"refresh_interval": "1h"}}
        if "max_merged_segment" in index_args.kwargs:
            settings["merge"] = {"policy": {"max_merged_segment": index_args.kwargs["max_merged_segment"]}}

        self.client.indices.create(
            index="test",
            settings=settings,
            mappings={
                "properties": {
                    # "_id": {"type": "integer"},
                    "text": {
                        "type": "dense_vector",
                        "dims": self.config.dataset.dim,
                        "index": True,
                        "similarity": "dot_product",
                        "element_type": ES_DATATYPES[index_args.quant],
                        "index_options": {
                            # use the same as element_type to make it simpler - we're not measuring rescoring yet
                            "type": ES_HNSW_TYPES[index_args.quant],
                            "m": index_args.m,
                            "ef_construction": index_args.ef_construction,
                        },
                    },
                    "tag": {"type": "integer"},
                }
            },
        )
        self.docs_in_segment = 0

    def stop(self, cleanup: bool):
        self.container.stop()
        if cleanup:
            self.container.remove()

    def commit(self):
        self.client.indices.refresh(index="test")
        if self.index_args.segments is not None:
            self.client.indices.forcemerge(
                index="test",
                max_num_segments=self.index_args.segments,
                wait_for_completion=True,
            )

    def index_batch(self, batch: list[Doc]) -> IndexResponse:
        actions = []
        for doc in batch:
            actions.append(
                {
                    "_op_type": "index",
                    "_index": "test",
                    "_source": {
                        "text": doc.embedding.tolist(),
                        "tag": doc.tag,
                    },
                    "_id": doc.id,
                }
            )
        start = time.perf_counter()
        helpers.bulk(self.client, actions)
        end = time.perf_counter()
        self.docs_in_segment += len(batch)
        if self.docs_in_segment >= self.index_args.kwargs.get("docs_per_segment", 1024):
            self.client.indices.refresh(index="test")
            self.docs_in_segment = 0
        return IndexResponse(client_latency=end - start)

    def search(self, search_params: SearchArgs, query: Query, top_k: int) -> SearchResponse:
        es_query = {
            "field": "text",
            "query_vector": query.embedding.tolist(),
            "k": top_k,
            "num_candidates": search_params.ef_search,
        }
        if search_params.filter_selectivity != 100:
            es_query["filter"] = {"terms": {"tag": [search_params.filter_selectivity]}}
        start = time.time_ns()
        response = self.client.search(index="test", knn=es_query, source=["_id"], size=top_k)
        end = time.time_ns()
        return SearchResponse(
            results=[DocScore(doc=int(doc["_id"]), score=doc["_score"]) for doc in response["hits"]["hits"]],
            client_latency=(end - start) / 1000000000.0,
        )
