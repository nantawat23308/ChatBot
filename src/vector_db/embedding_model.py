import os
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


class EmbeddingModel:
    def __init__(self, model_name: str = "amazon.titan-embed-text-v2:0"):
        self.__embedding_model: Embeddings = BedrockEmbeddings(
            model_id=model_name, region_name=os.getenv("AWS_REGION"), dimensions=1024
        )

    @property
    def embedding_model(self) -> Embeddings:
        return self.__embedding_model

    @embedding_model.setter
    def embedding_model(self, model: Embeddings):
        self.__embedding_model = model

    def get_embedding(self, text: str):
        return self.embedding_model.embed_query(text)

    def get_embeddings(self, texts: list[str]):
        return self.embedding_model.embed_documents(texts)


if __name__ == "__main__":
    embedder = EmbeddingModel()
    sample_text = "What is the capital of France?"
    embedding = embedder.get_embedding(sample_text)
    print(f"Embedding for '{sample_text}': {embedding}")
