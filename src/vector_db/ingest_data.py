from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vector_db.embedding_model import EmbeddingModel
from src.vector_db.loader_process import LoaderProcess
from src.vector_db.vector_store import QdrantVectorStoreWrapper
from src.logger import log
from typing import List, Callable
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

load_dotenv()


def get_vector_store(collection_name: str = "documents") -> VectorStore:
    """Get Qdrant Vector Store."""
    embedder = EmbeddingModel()
    vector_store_wrapper = QdrantVectorStoreWrapper(
        collection_name=collection_name, embeddings=embedder.embedding_model
    )
    vector_store = vector_store_wrapper.get_vector_store()
    return vector_store


def vector_db_add_document(
    vector_store: VectorStore,
    documents: List[Document],
    chunker: Callable = RecursiveCharacterTextSplitter,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Add document to Vector Store."""
    split = chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(
        documents
    )
    vector_store.add_documents(split)


def ingest_data(
    file_path: str | Path,
    collection_name: str = "documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Ingest data from a file into Qdrant vector store."""
    vector_store = get_vector_store(collection_name=collection_name)

    loader_process = LoaderProcess()
    documents = loader_process.loader(file_path)
    vector_db_add_document(
        vector_store=vector_store,
        documents=documents,
        chunker=RecursiveCharacterTextSplitter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def ingest_sparse(
    file_path: str | Path,
    collection_name: str = "documents_sparse",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Ingest sparse data from a JSON file into Qdrant vector store."""
    embedder = EmbeddingModel()
    vector_store_wrapper = QdrantVectorStoreWrapper(
        collection_name=collection_name, embeddings=embedder.embedding_model
    )
    vector_store_wrapper.retrieval_mode = "sparse"

    vector_store = vector_store_wrapper.get_vector_store()

    loader_process = LoaderProcess()
    documents = loader_process.loader(file_path)
    vector_db_add_document(
        vector_store=vector_store,
        documents=documents,
        chunker=RecursiveCharacterTextSplitter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def clear_collection(collection_name: str = "documents"):
    """Clear all documents from the specified collection in Qdrant vector store."""
    embedder = EmbeddingModel()
    vector_store_wrapper = QdrantVectorStoreWrapper(
        collection_name=collection_name, embeddings=embedder.embedding_model
    )
    vector_store_wrapper.clear_all_documents()
    log.debug(f"Cleared all documents from collection: {collection_name}")


if __name__ == "__main__":
    sample_file = "/home/nantawat/Desktop/my_project/chatbot/doc/1409.0473v7.pdf"  # Replace with your file path
    ingest_sparse(sample_file)
    # clear_collection()
