import logging
import time

import docker
from opensearchpy import OpenSearch, helpers

from hseb.core.config import Config, IndexArgs, QuantDatatype, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import DocScore, IndexResponse, SearchResponse
from hseb.engine.base import EngineBase

logger = logging.getLogger()

OS_DATATYPES = {
    QuantDatatype.FLOAT32: "float",
    QuantDatatype.INT8: "byte",
    QuantDatatype.INT1: "binary",
}


class OpenSearchEngine(EngineBase):
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
                "OPENSEARCH_JAVA_OPTS": f"-Xms{heap_size} -Xmx{heap_size}",
                "DISABLE_SECURITY_PLUGIN": "true",
                "bootstrap.memory_lock": "true",
            },
        )
        self._wait_for_logs(self.container, "Cluster health status changed from [YELLOW] to [GREEN]")
        self.client = OpenSearch(
            hosts=[{"host": "localhost", "port": 9200}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=30,
        )
        index_params = {
            "knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
        if "refresh_every" in index_args.kwargs:
            index_params["refresh_interval"] = "1h"
        if "max_merged_segment" in index_args.kwargs:
            index_params["merge"] = {"policy": {"max_merged_segment": index_args.kwargs["max_merged_segment"]}}
        field_params = {
            "type": "knn_vector",
            "dimension": self.config.dataset.dim,
            "space_type": "innerproduct",
            "data_type": "float",
            "method": {
                "name": "hnsw",
                "engine": index_args.kwargs.get("engine", "lucene"),
                "parameters": {
                    "m": index_args.m,
                    "ef_construction": index_args.ef_construction,
                },
            },
        }
        match index_args.quant:
            case QuantDatatype.FLOAT32:
                field_params["compression_level"] = "1x"
                # lucene or faiss
            case QuantDatatype.FLOAT16:
                field_params["compression_level"] = "2x"
                field_params["method"]["engine"] = "faiss"
            case QuantDatatype.INT8:
                field_params["compression_level"] = "4x"
                field_params["method"]["engine"] = "lucene"
        self.client.indices.create(
            index="test",
            body={
                "settings": {
                    "index": index_params,
                },
                "mappings": {
                    "properties": {
                        "text": field_params,
                        "tag": {"type": "integer"},
                    }
                },
            },
        )
        self.docs_in_segment = 0

    def stop(self, cleanup: bool):
        self.container.stop()
        if cleanup and self.container:
            self.container.remove(v=True)

    def commit(self):
        self.client.indices.refresh(index="test")
        if self.index_args.segments is not None:
            self.client.indices.forcemerge(
                index="test",
                params={
                    "max_num_segments": self.index_args.segments,
                    "wait_for_completion": True,
                },
            )

    def index_is_green(self) -> bool:
        response = self.client.cluster.health(index="test")
        return response["status"] == "green"

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

    def search(self, search_args: SearchArgs, query: Query, top_k: int) -> SearchResponse:
        knn_query = {
            "text": {
                "vector": query.embedding.tolist(),
                "k": top_k,
                "method_parameters": {
                    "ef_search": search_args.ef_search,
                },
            }
        }

        if search_args.filter_selectivity != 100:
            knn_query["text"]["filter"] = {"terms": {"tag": [search_args.filter_selectivity]}}

        search_body = {
            "size": top_k,
            "query": {"knn": knn_query},
            "_source": False,
        }

        start = time.time_ns()
        response = self.client.search(index="test", body=search_body)
        end = time.time_ns()
        return SearchResponse(
            results=[DocScore(doc=int(doc["_id"]), score=doc["_score"]) for doc in response["hits"]["hits"]],
            client_latency=(end - start) / 1000000000.0,
        )
