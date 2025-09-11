import logging
import time

import docker
import redis
from redis.commands.search.query import Query as RedisQuery

from hseb.core.config import Config, IndexArgs, QuantDatatype, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import DocScore, Response
from hseb.engine.base import EngineBase

logger = logging.getLogger()

REDIS_DATATYPES = {
    QuantDatatype.FLOAT32: "FLOAT32",
    QuantDatatype.FLOAT16: "FLOAT16",
}

REDIS_DISTANCE_METRIC = "COSINE"  # Cosine similarity


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

        # Create vector index
        datatype = REDIS_DATATYPES[index_args.quant]

        # Use HNSW algorithm for large datasets
        algorithm = "HNSW"
        algorithm_params = ["M", str(index_args.m), "EF_CONSTRUCTION", str(index_args.ef_construction)]

        try:
            # Create index with vector field using direct FT.CREATE command
            from redis.commands.search.field import VectorField, TextField, TagField

            self.client.ft("documents").create_index(
                [
                    VectorField(
                        "embedding",
                        algorithm,
                        {
                            "TYPE": datatype,
                            "DIM": self.config.dataset.dim,
                            "DISTANCE_METRIC": REDIS_DISTANCE_METRIC,
                            **dict(zip(algorithm_params[::2], algorithm_params[1::2])),
                        },
                    ),
                    TextField("text"),
                    TagField("tag", separator=","),  # Use comma separator for tags
                ]
            )
        except redis.ResponseError as e:
            if "Index already exists" not in str(e):
                raise e

        self.index_args = index_args
        return self

    def stop(self):
        if self.client:
            self.client.close()
        if self.container:
            self.container.stop()

    def commit(self):
        pass

    def index_batch(self, batch: list[Doc]):
        pipe = self.client.pipeline()

        for doc in batch:
            # Store document as hash with vector embedding
            doc_key = f"doc:{doc.id}"
            # Convert tag array to comma-separated string for Redis TagField
            tag_str = ",".join(map(str, doc.tag))

            # Use the same dtype as the index was created with
            if self.index_args.quant == QuantDatatype.FLOAT16:
                embedding_bytes = doc.embedding.astype("float16").tobytes()
            else:
                embedding_bytes = doc.embedding.astype("float32").tobytes()

            doc_data = {
                "text": doc.text,
                "embedding": embedding_bytes,
                "tag": tag_str,  # Store tags as comma-separated string
            }
            pipe.hset(doc_key, mapping=doc_data)

        pipe.execute()

    def search(self, search_params: SearchArgs, query: Query, top_k: int) -> Response:
        # Build KNN query - use the same dtype as the index was created with
        if self.index_args.quant == QuantDatatype.FLOAT16:
            query_vector = query.embedding.astype("float16").tobytes()
        else:
            query_vector = query.embedding.astype("float32").tobytes()

        # Create base query with AS clause for scoring
        base_query = f"(*)=>[KNN {top_k} @embedding $query_vector AS vector_score]"

        # Add filter if needed
        if search_params.filter_selectivity != 100:
            # Filter by tag using TagField exact match syntax with curly brackets
            filter_query = f"(@tag:{{{search_params.filter_selectivity}}})"
            base_query = f"({filter_query})=>[KNN {top_k} @embedding $query_vector AS vector_score]"

        # Use ef_search parameter for HNSW search
        search_params_dict = {"query_vector": query_vector}

        start = time.time_ns()

        results = self.client.ft("documents").search(
            RedisQuery(base_query)
            .sort_by("vector_score", asc=False)  # Sort by score descending (best matches first)
            .return_fields("vector_score", "text", "tag")
            .dialect(2)
            .paging(0, top_k),
            query_params=search_params_dict,
        )
        end = time.time_ns()

        # Convert results to DocScore objects
        doc_scores = []
        for i, result in enumerate(results.docs):
            # Extract document ID from key (doc:123 -> 123)
            # Handle both string and bytes for document ID
            doc_key = result.id
            doc_id = int(doc_key.split(":")[1])
            # Redis returns similarity score in vector_score field
            score = float(result.vector_score)
            doc_scores.append(DocScore(doc=doc_id, score=score))

        return Response(
            results=doc_scores,
            client_latency=(end - start) / 1000000000.0,
        )
