import os
import hashlib
from typing import List

from document import DocumentProcessor

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class AgenticRAG:
    def __init__(self):
        self.emdeddings = self._setup_embeddings()
        self.vector_store = self._setup_vector_store()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
        self.doc_processor = DocumentProcessor()
        self.document_index = {}

    def _setup_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    def _setup_vector_store(self):
        os.makedirs("./chroma_db", exist_ok=True)
        return Chroma(
            embedding_function=self.emdeddings,
            persist_directory="./chroma_db",
        )

    def add_document(self, file_path: str):
        try:
            file_hash = self._get_file_hash(str(file_path))
            if file_hash in self.document_index:
                print(f"Document {file_path} already indexed.")

            docs = self.doc_processor.load_document(file_path)
            if not docs:
                print(f"No documents found in {file_path}.")
                return

            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                print(f"No chunks created from {file_path}.")
                return

            self.vector_store.add_documents(chunks)
        except Exception as e:
            print(f"Error adding document {file_path}: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error in similarity search with score: {e}")
            return []
