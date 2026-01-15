from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_docling import DoclingLoader
from src.vector_db.embedding_model import EmbeddingModel
from langchain_core.documents import Document


def chunk_semantic(file_path: str) -> list[Document]:
    """Chunk a text file into smaller pieces.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list[str]: A list of text chunks.
    """
    # Load the text file
    loader = DoclingLoader(file_path)
    documents = loader.load()
    embedder = EmbeddingModel()
    semantic_chunker = SemanticChunker(
        embedder.embedding_model, breakpoint_threshold_type="percentile"
    )
    semantic_chunks = semantic_chunker.split_documents(documents)
    return semantic_chunks


def chunk_text_file(
    file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Chunk a text file into smaller pieces.

    Args:
        file_path (str): The path to the text file.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.

    Returns:
        list[str]: A list of text chunks.
    """
    # Load the text file
    loader = DoclingLoader(file_path)
    documents = loader.load()

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    # Split the documents into chunks
    output_chunks = text_splitter.split_documents(documents)
    return output_chunks


if __name__ == "__main__":
    file_path = "/home/nantawat/Desktop/my_project/chatbot/doc/1409.0473v7.pdf"
    chunks = chunk_text_file(file_path)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")
