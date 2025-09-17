import logging
import time

import docker
import psycopg2
from psycopg2.extras import execute_values

from hseb.core.config import Config, IndexArgs, QuantDatatype, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import DocScore, IndexResponse, SearchResponse
from hseb.engine.base import EngineBase

logger = logging.getLogger()

POSTGRES_DATATYPES = {
    QuantDatatype.FLOAT32: "vector",
    QuantDatatype.FLOAT16: "halfvec",
    QuantDatatype.INT1: "bit",
}

POSTGRES_INDEX_OPS = {
    QuantDatatype.FLOAT32: "vector_ip_ops",
    QuantDatatype.FLOAT16: "halfvec_ip_ops",
    QuantDatatype.INT1: "bit_hamming_ops",
}


class PostgresEngine(EngineBase):
    def __init__(self, config: Config):
        self.config = config
        self.connection = None
        self.container = None

    def start(self, index_args: IndexArgs):
        # Check for unsupported quantization types
        if index_args.quant == QuantDatatype.INT8:
            raise ValueError("Postgres with pgvector does not support INT8 quantization")
        if index_args.segments is not None:
            raise ValueError("Postgres with pgvector cannot set number of segments")

        docker_client = docker.from_env()

        # PostgreSQL configuration for better performance
        shared_buffers = index_args.kwargs.get("shared_buffers", "2GB")
        work_mem = index_args.kwargs.get("work_mem", "16MB")
        maintenance_work_mem = index_args.kwargs.get("maintenance_work_mem", "512MB")

        self.container = docker_client.containers.run(
            image=self.config.image,
            ports={"5432/tcp": 5432},
            detach=True,
            shm_size="4g",  # needed for buffers
            environment={
                "POSTGRES_DB": "benchmark",
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "postgres",
                "POSTGRES_HOST_AUTH_METHOD": "trust",
            },
            command=[
                "postgres",
                "-c",
                f"shared_buffers={shared_buffers}",
                "-c",
                f"work_mem={work_mem}",
                "-c",
                f"maintenance_work_mem={maintenance_work_mem}",
                "-c",
                "max_connections=100",
                "-c",
                "random_page_cost=1.1",
                "-c",
                "effective_cache_size=1GB",
                "-c",
                "max_parallel_maintenance_workers=7",
            ],
        )
        self._wait_for_logs(self.container, "database system is ready to accept connections")

        # Wait a bit more to ensure PostgreSQL is fully ready
        time.sleep(2)

        # Connect to database
        self.connection = psycopg2.connect(
            host="localhost", port=5432, database="benchmark", user="postgres", password="postgres"
        )
        self.connection.autocommit = True

        # Create pgvector extension and table
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            datatype = POSTGRES_DATATYPES[index_args.quant]
            cursor.execute(f"""
                CREATE UNLOGGED TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    embedding {datatype}({self.config.dataset.dim}),
                    tag INTEGER[]
                );
            """)
            cursor.execute("SET max_parallel_maintenance_workers = 7;")
            cursor.execute("SET client_min_messages = DEBUG;")

            time.sleep(5)
            # Create HNSW index with appropriate operator class
            ops_class = POSTGRES_INDEX_OPS[index_args.quant]
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING hnsw (embedding {ops_class})
                WITH (m = %s, ef_construction = %s);
            """,
                (index_args.m, index_args.ef_construction),
            )

            # Create GIN index on tag array for efficient filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS documents_tag_idx ON documents USING gin (tag);")
            time.sleep(5)

        self.index_args = index_args  # Store for later use
        return self

    def stop(self, cleanup: bool):
        if self.connection:
            self.connection.close()
        if self.container:
            self.container.stop()
        if cleanup and self.container is not None:
            self.container.remove()

    def commit(self):
        # Just optimize table
        with self.connection.cursor() as cursor:
            cursor.execute("VACUUM ANALYZE documents;")

    def index_batch(self, batch: list[Doc]) -> IndexResponse:
        # Prepare data for batch insert
        data = [(doc.id, doc.embedding.tolist(), doc.tag) for doc in batch]

        with self.connection.cursor() as cursor:
            start = time.perf_counter()
            execute_values(
                cursor,
                "INSERT INTO documents (id, embedding, tag) VALUES %s ON CONFLICT (id) DO NOTHING",
                data,
                template=None,
                page_size=len(data),
            )
            end = time.perf_counter()
            return IndexResponse(client_latency=end - start)

    def search(self, search_params: SearchArgs, query: Query, top_k: int) -> SearchResponse:
        # Build the query - use appropriate distance function with proper type casting
        if self.index_args.quant == QuantDatatype.INT1:
            # For bit vectors, use Hamming distance
            sql = """
                SELECT id, embedding <~> %s::bit AS distance
                FROM documents
            """
        elif self.index_args.quant == QuantDatatype.FLOAT16:
            # For halfvec, cast to halfvec type
            sql = """
                SELECT id, embedding <#> %s::halfvec AS distance  
                FROM documents
            """
        else:
            # For float32 vectors, cast to vector type
            sql = """
                SELECT id, embedding <#> %s::vector AS distance  
                FROM documents
            """

        params = [query.embedding.tolist()]

        # Add filtering if needed
        if search_params.filter_selectivity != 100:
            sql += " WHERE %s = ANY(tag)"
            params.append(search_params.filter_selectivity)

        sql += " ORDER BY distance LIMIT %s"
        params.append(top_k)

        with self.connection.cursor() as cursor:
            cursor.execute("SET hnsw.ef_search = %s", (search_params.ef_search,))
            cursor.execute("SET hnsw.iterative_scan = strict_order")
            start = time.time_ns()
            cursor.execute(sql, params)
            results = cursor.fetchall()
        end = time.time_ns()

        # Convert to DocScore objects
        if self.index_args.quant == QuantDatatype.INT1:
            # For Hamming distance, convert to similarity score (lower distance = higher similarity)
            doc_scores = [DocScore(doc=row[0], score=1.0 / (1.0 + row[1])) for row in results]
        else:
            # For inner product, convert negative inner product to positive score
            doc_scores = [DocScore(doc=row[0], score=-row[1]) for row in results]

        return SearchResponse(
            results=doc_scores,
            client_latency=(end - start) / 1000000000.0,
        )
