import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import  List, Dict, Any
from ..retriever.document_retriever import DocumentRetriever


class FAISSRetriever(DocumentRetriever):
    def __init__(self, embedding_dim: int = 384):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        embeddings = [self.embedding_model.encode(doc) for doc in documents]
        embeddings_np = np.array(embeddings).astype('float32')

        self.index.add(embeddings_np)
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas or [{} for _ in documents])

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = np.array([self.embedding_model.encode(query_text)]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx]
                })

        return results

    def persist(self, persist_path: str) -> None:
        faiss.write_index(self.index, f"{persist_path}/faiss_index.bin")

    def load(self, persist_path: str) -> None:
        self.index = faiss.read_index(f"{persist_path}/faiss_index.bin")
