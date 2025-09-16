import logging
import time

import docker
import redis
from redis.commands.search.query import Query as RedisQuery
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from hseb.core.config import Config, IndexArgs, QuantDatatype, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import DocScore, IndexResponse, SearchResponse
from hseb.engine.base import EngineBase

logger = logging.getLogger()

REDIS_DATATYPES = {
    QuantDatatype.FLOAT32: "FLOAT32",
    QuantDatatype.FLOAT16: "FLOAT16",
    QuantDatatype.INT8: "INT8",
}


class RedisEngine(EngineBase):
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.container = None

    def start(self, index_args: IndexArgs):
        # Check for unsupported quantization types
        if index_args.quant not in REDIS_DATATYPES:
            raise ValueError(
                f"Redis does not support {index_args.quant} quantization. Supported types: {list(REDIS_DATATYPES.keys())}"
            )
        if index_args.segments is not None:
            raise ValueError("Redis cannot set number of segments")

        docker_client = docker.from_env()

        # Redis configuration
        maxmemory = index_args.kwargs.get("maxmemory", "2gb")
        maxmemory_policy = index_args.kwargs.get("maxmemory_policy", "allkeys-lru")

        self.container = docker_client.containers.run(
            image=self.config.image,
            ports={"6379/tcp": 6379},
            detach=True,
            command=["redis-server", "--maxmemory", maxmemory, "--maxmemory-policy", maxmemory_policy],
        )
        self._wait_for_logs(self.container, "Ready to accept connections")

        # Wait a bit more to ensure Redis is fully ready
        time.sleep(2)

        # Connect to Redis
        self.client = redis.Redis(host="localhost", port=6379, decode_responses=False)

        # Create index with vector field using direct FT.CREATE command

        self.client.ft("documents").create_index(
            fields=[
                VectorField(
                    name="embedding",
                    algorithm="HNSW",
                    attributes={
                        "TYPE": REDIS_DATATYPES[index_args.quant],
                        "DIM": self.config.dataset.dim,
                        "DISTANCE_METRIC": "IP",
                        "M": index_args.m,
                        "EF_CONSTRUCTION": index_args.ef_construction,
                    },
                ),
                TagField("tag", separator=","),  # Use comma separator for tags
            ],
            definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH),
        )

        self.index_args = index_args
        return self

    def stop(self, cleanup: bool):
        if self.client:
            self.client.close()
        if self.container:
            self.container.stop()
        if cleanup:
            self.container.remove()

    def commit(self):
        pass

    def index_batch(self, batch: list[Doc]) -> IndexResponse:
        pipe = self.client.pipeline()

        for doc in batch:
            # Use the same dtype as the index was created with
            match self.index_args.quant:
                case QuantDatatype.FLOAT16:
                    embedding_bytes = doc.embedding.astype("float16").tobytes()
                case QuantDatatype.FLOAT32:
                    embedding_bytes = doc.embedding.astype("float32").tobytes()
                case QuantDatatype.INT8:
                    embedding_bytes = doc.embedding.astype("int8").tobytes()

            pipe.hset(
                f"doc:{doc.id}",
                mapping={
                    "embedding": embedding_bytes,
                    "tag": ",".join(map(str, doc.tag)),  # Store tags as comma-separated string
                },
            )
        start = time.perf_counter()
        pipe.execute()
        end = time.perf_counter()
        return IndexResponse(client_latency=end - start)

    def search(self, search_params: SearchArgs, query: Query, top_k: int) -> SearchResponse:
        # Build KNN query - use the same dtype as the index was created with
        match self.index_args.quant:
            case QuantDatatype.FLOAT16:
                embedding_bytes = query.embedding.astype("float16").tobytes()
            case QuantDatatype.FLOAT32:
                embedding_bytes = query.embedding.astype("float32").tobytes()
            case QuantDatatype.INT8:
                embedding_bytes = query.embedding.astype("int8").tobytes()

        # Create base query with AS clause for scoring
        base_query = f"(*)=>[KNN {top_k} @embedding $query_vector AS vector_score]"

        # Add filter if needed
        if search_params.filter_selectivity != 100:
            # Filter by tag using TagField exact match syntax with curly brackets
            base_query = f"(@tag:{{{search_params.filter_selectivity}}})=>[KNN {top_k} @embedding $query_vector AS vector_score]"

        # Use ef_search parameter for HNSW search

        start = time.time_ns()

        results = self.client.ft("documents").search(
            RedisQuery(base_query)
            .sort_by("vector_score", asc=False)  # Sort by score descending (best matches first)
            .return_fields("vector_score")
            .dialect(2)
            .paging(0, top_k),
            query_params={"query_vector": embedding_bytes},
        )
        end = time.time_ns()

        # Convert results to DocScore objects
        doc_scores = []
        for result in results.docs:
            # Extract document ID from key (doc:123 -> 123)
            doc_id = int(result.id.split(":")[1])
            # Redis returns similarity score in vector_score field
            score = float(result.vector_score)
            doc_scores.append(DocScore(doc=doc_id, score=score))

        return SearchResponse(
            results=doc_scores,
            client_latency=(end - start) / 1000000000.0,
        )
