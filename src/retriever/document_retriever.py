import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import PyPDF2


class DocumentRetriever(ABC):
    """
    Base class for document retrieval from vector databases.
    """

    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Add documents and their embeddings to the vector store.

        :param documents: List of text documents to be added.
        :param metadatas: Optional metadata associated with each document.
        """
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar documents based on the query text.

        :param query_text: The query text to search for.
        :param top_k: Number of top results to return.
        :return: A list of dictionaries containing retrieved documents and metadata.
        """
        pass

    @abstractmethod
    def persist(self, persist_path: str) -> None:
        """
        Persist the vector store to disk.

        :param persist_path: Path to save the vector store.
        """
        pass

    @abstractmethod
    def load(self, persist_path: str) -> None:
        """
        Load the vector store from disk.

        :param persist_path: Path to load the vector store from.
        """
        pass

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extracts text from a PDF file and returns a list of text content per page.

        :param pdf_path: Path to the PDF file.
        :return: List of strings, each representing text extracted from a page.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text_list = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_list.append(text.strip())

        print(f"Extracted {len(text_list)} pages of text from PDF: {pdf_path}")
        return text_list
