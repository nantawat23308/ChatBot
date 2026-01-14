from langchain_docling import DoclingLoader
from langchain_community.document_loaders import DiffbotLoader, TextLoader
from pathlib import Path
from typing import List


class LoaderProcess:
    def __init__(self):
        pass

    def loader(self, file_path: str | Path) -> List:
        file_path = Path(file_path)
        if file_path.suffix in [".txt", ".md"]:
            loader = self.text_loader(str(file_path))
        elif file_path.suffix in [".json", ".pdf"]:
            loader = self.docling_loader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        documents = loader.load()
        return documents

    def loader_type(self, file_path: str | Path, loader_type: str, **kwargs) -> List:
        file_path = Path(file_path)
        if loader_type == "text":
            loader = self.text_loader(str(file_path))
        elif loader_type == "docling":
            loader = self.docling_loader(str(file_path))
        elif loader_type == "diffbot":
            token = kwargs.get("token", "")
            loader = self.diffbot_loader(str(file_path), token)
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")
        documents = loader.load()
        return documents

    def directory_loader(self, dir_path: str | Path) -> List:
        dir_path = Path(dir_path)
        all_documents = []
        for file in dir_path.iterdir():
            if file.is_file():
                documents = self.loader(file)
                all_documents.extend(documents)
        return all_documents

    def docling_loader(self, url: str):
        loader = DoclingLoader(url)
        return loader

    def diffbot_loader(self, url: str, token: str):
        loader = DiffbotLoader(url, diffbot_token=token)
        return loader

    def text_loader(self, url: str):
        loader = TextLoader(url)
        return loader
