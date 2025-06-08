import os
from typing import List
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = {".pdf", ".txt", ".docx"}

    def load_document(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            if ext == ".pdf":
                return self._load_pdf(file_path)
            elif ext == ".txt":
                return self._load_txt(file_path)
            elif ext == ".docx":
                return self._load_docx(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def _load_pdf(self, file_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata.update(
                    {
                        "file_path": file_path,
                        "file_type": "pdf",
                        "processed_at": datetime.now().isoformat(),
                    }
                )
            return docs
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []

    def _load_txt(self, file_path: str) -> List[Document]:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata.update(
                    {
                        "file_path": file_path,
                        "file_type": "txt",
                        "processed_at": datetime.now().isoformat(),
                    }
                )
            return docs
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return []

    def _load_docx(self, file_path: str) -> List[Document]:
        try:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata.update(
                    {
                        "file_path": file_path,
                        "file_type": "docx",
                        "processed_at": datetime.now().isoformat(),
                    }
                )
            return docs
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
            return []
