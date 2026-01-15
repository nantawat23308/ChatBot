import os

from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.logger import log
from src.vector_db.embedding_model import EmbeddingModel
from src.vector_db.vector_store import QdrantVectorStoreWrapper

load_dotenv()


def search_similar_documents(
    query: str, collection_name: str = "documents", top_k: int = 5
):
    """Search for similar documents in Qdrant vector store based on the query."""
    embedder = EmbeddingModel()
    vector_store_wrapper = QdrantVectorStoreWrapper(
        collection_name=collection_name, embeddings=embedder.embedding_model
    )
    vector_store = vector_store_wrapper.get_vector_store()

    query_embedding = embedder.get_embedding(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=top_k)

    log.debug(f"Found {len(results)} similar documents for query: '{query}'")
    return results


def search_with_score(query: str, collection_name: str = "documents", top_k: int = 5):
    """Search for similar documents with scores in Qdrant vector store based on the query."""
    embedder = EmbeddingModel()
    vector_store_wrapper = QdrantVectorStoreWrapper(
        collection_name=collection_name, embeddings=embedder.embedding_model
    )
    vector_store = vector_store_wrapper.get_vector_store()

    results_with_scores = vector_store.similarity_search_with_score(
        query=query, k=top_k
    )

    log.debug(
        f"Found {len(results_with_scores)} similar documents with scores for query: '{query}'"
    )
    return results_with_scores


def vector_store_as_retriever(collection_name: str = "documents", top_k: int = 5):
    """Get the Qdrant vector store as a LangChain retriever."""
    embedder = EmbeddingModel()
    vector_store_wrapper = QdrantVectorStoreWrapper(
        collection_name=collection_name, embeddings=embedder.embedding_model
    )
    vector_store = vector_store_wrapper.get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )
    return retriever


def rerank_retriever(
    collection_name: str = "documents", top_k: int = 5, top_k_first_retriever: int = 15
):
    """Rerank documents from a retriever based on similarity to the query."""
    retriever = vector_store_as_retriever(
        collection_name=collection_name, top_k=top_k_first_retriever
    )
    model = HuggingFaceCrossEncoder(model_name=os.getenv("RERANKER_MODEL"))
    reranker = CrossEncoderReranker(model=model, top_n=top_k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )
    return compression_retriever


if __name__ == "__main__":
    pass
