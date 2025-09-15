import docker
from hseb.core.config import Config, IndexArgs, QuantDatatype, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import DocScore, IndexResponse, SearchResponse
from hseb.engine.base import EngineBase
from qdrant_client import QdrantClient
import time
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    ScalarQuantizationConfig,
    ScalarType,
    BinaryQuantizationConfig,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    CollectionStatus,
    OptimizersConfigDiff,
    HnswConfigDiff,
    ScalarQuantization,
    BinaryQuantization,
)

QDRANT_DATATYPES = {
    QuantDatatype.INT8: ScalarQuantization(scalar=ScalarQuantizationConfig(type=ScalarType.INT8)),
    QuantDatatype.INT1: BinaryQuantization(binary=BinaryQuantizationConfig()),
}


class Qdrant(EngineBase):
    def __init__(self, config: Config):
        self.config = config

    def index_batch(self, batch: list[Doc]) -> IndexResponse:
        points = [PointStruct(id=doc.id, vector=doc.embedding.tolist(), payload={"tag": doc.tag}) for doc in batch]
        start = time.perf_counter()
        self.client.upsert(collection_name="test", points=points)
        end = time.perf_counter()
        return IndexResponse(client_latency=end - start)

    def commit(self):
        is_green = False
        attempts = 0
        while not is_green and attempts < 60:
            status = self.client.get_collection(collection_name="test")
            if status.status == CollectionStatus.GREEN:
                is_green = True
            else:
                attempts += 1
                time.sleep(1)
        if not is_green:
            raise Exception("collection stuck at non-green status")

    def search(self, search_params: SearchArgs, query: Query, top_k: int) -> SearchResponse:
        def create_filter():
            if search_params.filter_selectivity == 100:
                return None
            else:
                return Filter(must=FieldCondition(key="tag", match=MatchValue(value=search_params.filter_selectivity)))

        start = time.time_ns()
        response = self.client.query_points(
            collection_name="test",
            query=query.embedding,
            query_filter=create_filter(),
            search_params=SearchParams(
                hnsw_ef=search_params.ef_search,
                indexed_only=True,  # to avoid fullscans over memtable
                exact=False,
            ),
            limit=top_k,
        )
        end = time.time_ns()
        return SearchResponse(
            results=[DocScore(point.id, point.score) for point in response.points],
            client_latency=(end - start) / 1000000000.0,
        )

    def start(self, index_args: IndexArgs):
        docker_client = docker.from_env()
        self.container = docker_client.containers.run(
            image=self.config.image,
            ports={"6333/tcp": 6333},
            detach=True,
        )
        self._wait_for_logs(self.container, "Qdrant gRPC listening")

        self.client = QdrantClient(host="localhost")
        self.client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(
                size=self.config.dataset.dim,
                distance=Distance.DOT,
                on_disk=index_args.kwargs.get("original_vectors_on_disk", None),
                datatype=None,  # it's about only the raw vectors, not the index
            ),
            quantization_config=QDRANT_DATATYPES.get(index_args.quant, None),
            hnsw_config=HnswConfigDiff(
                m=index_args.m,
                ef_construct=index_args.ef_construction,
                on_disk=index_args.kwargs.get("hnsw_on_disk", None),
            ),
            optimizers_config=OptimizersConfigDiff(
                max_segment_size=index_args.kwargs.get("max_segment_size_kb", None),
                default_segment_number=index_args.kwargs.get("default_segment_number", index_args.segments),
            ),
        )
        return self

    def stop(self, cleanup: bool):
        self.container.stop()
        if cleanup:
            self.container.remove()
