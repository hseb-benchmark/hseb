import logging
import time
from uuid import UUID
import uuid

import docker
import weaviate
from weaviate.classes.config import Configure, VectorDistances, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.data import DataObject


from hseb.core.config import Config, IndexArgs, QuantDatatype, SearchArgs
from hseb.core.dataset import Doc, Query
from hseb.core.response import DocScore, Response
from hseb.engine.base import EngineBase

logger = logging.getLogger()

WEAVIATE_DATATYPES = {
    QuantDatatype.FLOAT32: None,  # Default precision
    QuantDatatype.INT8: Configure.VectorIndex.Quantizer.sq(),  # Scalar quantization for int8
    QuantDatatype.INT1: Configure.VectorIndex.Quantizer.bq(),  # Binary quantization for int1
}


class WeaviateEngine(EngineBase):
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.container = None
        self.collection_name = "TestCollection"

    def start(self, index_args: IndexArgs):
        docker_client = docker.from_env()

        self.container = docker_client.containers.run(
            image=self.config.image,
            ports={"8080/tcp": 8080, "50051/tcp": 50051},
            detach=True,
            environment={"PERSISTENCE_DATA_PATH": "/var/lib/weaviate"},
        )
        self._wait_for_logs(self.container, "Serving weaviate at")

        # Wait a bit more to ensure Weaviate is fully ready
        time.sleep(3)

        # Connect to Weaviate
        self.client = weaviate.connect_to_local(port=8080)

        # Configure quantizer based on quantization type
        quantizer = WEAVIATE_DATATYPES.get(index_args.quant, None)

        # Create collection (using deprecated parameter for now to test basic functionality)
        self.client.collections.create(
            name=self.collection_name,
            vector_config=Configure.Vectors.self_provided(
                name="self",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.DOT,
                    ef_construction=index_args.ef_construction,
                    max_connections=index_args.m,
                ),
                quantizer=quantizer,
            ),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="tag", data_type=DataType.INT_ARRAY),
            ],
        )

        self.index_args = index_args
        self.uuid_docid_cache: dict[UUID, int] = {}
        return self

    def stop(self):
        if self.client:
            self.client.close()
        if self.container:
            self.container.stop()

    def commit(self):
        pass

    def index_batch(self, batch: list[Doc]):
        collection = self.client.collections.get(self.collection_name)

        # Prepare objects for batch insertion
        objects = []
        for doc in batch:
            key = uuid.uuid4()
            self.uuid_docid_cache[key] = doc.id
            obj = DataObject(
                properties={
                    "text": doc.text,
                    "tag": doc.tag,
                },
                uuid=key,
                vector=doc.embedding.tolist(),
            )
            objects.append(obj)

        # Batch insert
        collection.data.insert_many(objects)

    def search(self, search_params: SearchArgs, query: Query, top_k: int) -> Response:
        collection = self.client.collections.get(self.collection_name)

        # Create filter if needed
        filters = None
        if search_params.filter_selectivity != 100:
            filters = Filter.by_property("tag").equal(search_params.filter_selectivity)

        start = time.time_ns()

        # Perform vector search with dot product similarity
        response = collection.query.near_vector(
            near_vector=query.embedding.tolist(),
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )

        end = time.time_ns()

        # Convert results to DocScore objects
        doc_scores = []
        for obj in response.objects:
            id = self.uuid_docid_cache[obj.uuid]
            doc_scores.append(DocScore(doc=id, score=-obj.metadata.distance))

        return Response(
            results=doc_scores,
            client_latency=(end - start) / 1000000000.0,
        )
