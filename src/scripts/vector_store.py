import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
from src.logger import log


class QdrantVectorStoreWrapper:
    def __init__(self, collection_name: str, embeddings=None):
        self.collection_name = collection_name
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(self.url)
        self.__embeddings = embeddings

    @property
    def embedding(self):
        return self.__embeddings

    @embedding.setter
    def embedding(self, embeddings):
        self.__embeddings = embeddings

    def get_vector_store(self):
        """Langchain Vector Store."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )

    def from_existing_collection(self):
        """Load existing Qdrant collection as Langchain Vector Store."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding(),
        )

    def from_documents(self, documents):
        """Create Langchain Vector Store from documents."""
        QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        ).from_documents(
            documents=documents,
            embedding=self.embedding,
            force_recreate=True,
        )

    def clear_all_documents(self):
        """Completely clears the collection."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        log.debug("Cleared all documents in the Qdrant collection.")

    def clear_by_metadata(self, filter_key: str, filter_value: str):
        """Clears documents based on metadata filter."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointsSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=f"metadata.{filter_key}",
                            match=models.MatchValue(value=filter_value)
                        )
                    ]
                )
            )
        )
        log.debug(f"Cleared documents with {filter_key}={filter_value} in the Qdrant collection.")
