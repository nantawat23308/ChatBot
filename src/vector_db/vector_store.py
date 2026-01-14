import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from src.logger import log


class QdrantVectorStoreWrapper:
    def __init__(self, collection_name: str, embeddings=None):
        self.collection_name = collection_name
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(self.url)

        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self._size_of_vector = 1024
        self._model_distance = models.Distance.COSINE

        self.__embeddings = embeddings
        self.__retrieval_mode = RetrievalMode.DENSE

    @property
    def embedding(self):
        return self.__embeddings

    @embedding.setter
    def embedding(self, embeddings):
        self.__embeddings = embeddings

    @property
    def retrieval_mode(self):
        return self.__retrieval_mode

    @retrieval_mode.setter
    def retrieval_mode(self, mode: RetrievalMode):
        self.__retrieval_mode = mode

    def sparse_vector_config(self):
        """Configuration for sparse vector embeddings."""
        return {
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }

    def get_vector_store(self):
        """Langchain Vector Store."""
        self.create_collection_not_exist()
        if self.retrieval_mode == RetrievalMode.DENSE:
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding,
            )
        elif self.retrieval_mode == RetrievalMode.SPARSE:
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode=self.retrieval_mode,
                sparse_vector_name="sparse",
            )
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode=self.retrieval_mode,
                vector_name="dense",
                sparse_vector_name="sparse",
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {self.retrieval_mode}")

    def create_collection_not_exist(self):
        """Create collection on Qdrant if it does not exist."""
        if not self.client.collection_exists(self.collection_name):
            if self.retrieval_mode == RetrievalMode.DENSE:
                self.create_collection_dense()
            elif self.retrieval_mode == RetrievalMode.SPARSE:
                self.create_collection_sparse()
            elif self.retrieval_mode == RetrievalMode.HYBRID:
                self.create_collection_sparse()
            log.debug(f"Created collection: {self.collection_name}")

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
            log.debug(f"Deleted existing collection: {self.collection_name}")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self._size_of_vector, distance=self._model_distance
            ),
        )
        log.debug("Cleared all documents in the Qdrant collection.")

    def create_collection_dense(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self._size_of_vector, distance=self._model_distance
            ),
        )

    def create_collection_sparse(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._size_of_vector, distance=self._model_distance
                )
            },
            sparse_vectors_config=self.sparse_vector_config(),
        )

    def clear_by_metadata(self, filter_key: str, filter_value: str):
        """Clears documents based on metadata filter."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointsSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=f"metadata.{filter_key}",
                            match=models.MatchValue(value=filter_value),
                        )
                    ]
                )
            ),
        )
        log.debug(
            f"Cleared documents with {filter_key}={filter_value} in the Qdrant collection."
        )
