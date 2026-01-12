from src.scripts.embedding_model import EmbeddingModel
from pathlib import Path
from src.scripts.loader_process import LoaderProcess
from src.scripts.vector_store import QdrantVectorStoreWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
from src.logger import log

def ingest_data(file_path: str | Path, collection_name: str = "documents"):
    """Ingest data from a file into Qdrant vector store."""
    loader_process = LoaderProcess()
    documents = loader_process.loader(file_path)

    embedder = EmbeddingModel()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    vector_store = QdrantVectorStoreWrapper(
        collection_name=collection_name,
        embeddings=embedder.embedding_model
    )
    # vector_store.clear_all_documents()
    # vector_store.get_vector_store().add_documents(splits)
    vector_store.from_documents(splits)

    log.debug(f"Saved {len(splits)} documents")


if __name__ == "__main__":
    sample_file = "/home/nantawat/Desktop/my_project/chatbot/doc/1409.0473v7.pdf"  # Replace with your file path
    ingest_data(sample_file)

