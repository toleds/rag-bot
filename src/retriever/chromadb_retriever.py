import chromadb
import os
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import  List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..retriever.document_retriever import DocumentRetriever

class ChromaDBRetriever(DocumentRetriever):
    def __init__(self, persist_directory: str = "./chroma_data"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        print("Creating collection...")
        self.collection = self.client.get_or_create_collection(name="documents")
        print("Collection created:", self.collection)
        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        print(f"Contents of {persist_directory}: {os.listdir(persist_directory)}")

    def add_documents(self, source: Any, metadatas: List[Dict[str, Any]] = None, source_type: str = "list") -> None:
        """
        Adds documents to the ChromaDB collection.

        :param source: List of documents or path to PDF file.
        :param metadatas: List of metadata dictionaries.
        :param source_type: Type of source ("list" or "pdf").
        """

        if source_type == "list":
            documents = source
        elif source_type == "pdf":
            documents = self.extract_text_from_pdf(source)
        else:
            raise ValueError("Unsupported source type. Use 'list' or 'pdf'.")

        embeddings = [self.embedding_model.encode(doc) for doc in documents]
        print(f"Generated embeddings: {embeddings}")

        metadatas = metadatas or [{} for _ in documents]

        # Generate unique ids for each document (e.g., using index or a custom ID generation)
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Add documents to the ChromaDB collection
        self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
        print("Documents successfully added to the collection.")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query_text)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        # document_embeddings = [self.embedding_model.encode(doc) for doc in results["documents"]]

        # # Reshape the query_embedding to 2D, because cosine_similarity expects 2D arrays
        # query_embedding_reshaped = np.expand_dims(query_embedding, axis=0)  # Shape: (1, n_features)
        # document_embeddings_reshaped = np.array(document_embeddings)  # (n_docs, n_features)
        # # Compute cosine similarity
        # similarities = cosine_similarity(query_embedding_reshaped, document_embeddings_reshaped)[0]  # (n_docs,)
        # # print(similarities)

        return [{"document": doc, "metadata": meta} for doc, meta in zip(results["documents"], results["metadatas"])]

    def persist(self, persist_path: str) -> None:
        print("Persistence is automatic with ChromaDB when using persist_directory.")

    def load(self, persist_path: str) -> None:
        self.client = chromadb.Client(Settings(persist_directory=persist_path))
        self.collection = self.client.get_collection(name="documents")
