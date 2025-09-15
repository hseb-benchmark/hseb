import logging
import time

import docker
from elasticsearch import Elasticsearch, helpers

from hseb.core.config import Config, IndexArgs, QuantDatatype, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import DocScore, Response
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
        self.client.indices.create(
            index="test",
            settings={
                "index": {
                    "refresh_interval": "1h",
                    "merge": {"policy": {"max_merged_segment": index_args.kwargs.get("max_merged_segment", "128mb")}},
                },
            },  # we control segment size
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

    def index_batch(self, batch: list[Doc]):
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
        helpers.bulk(self.client, actions)
        self.docs_in_segment += len(batch)
        if self.docs_in_segment >= self.index_args.kwargs.get("docs_per_segment", 1024):
            self.client.indices.refresh(index="test")
            self.docs_in_segment = 0

    def search(self, search_params: SearchArgs, query: Query, top_k: int) -> Response:
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
        return Response(
            results=[DocScore(doc=int(doc["_id"]), score=doc["_score"]) for doc in response["hits"]["hits"]],
            client_latency=(end - start) / 1000000000.0,
        )
