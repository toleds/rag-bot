from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

class Retriever:
    def __init__(self, vector_store: str, persist_directory: str = "./vector_data", embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Universal Retriever that can work with either Chroma or FAISS.
        :param vector_store: The type of vector store ('chroma' or 'faiss').
        :param persist_directory: Directory to store the vector store data.
        :param embedding_model_name: The name of the embedding model to use.
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Decide which vector store to use (Chroma or FAISS)
        if vector_store == 'chroma':
            self.vector_store = self._get_chroma_instance(persist_directory)
        elif vector_store == 'faiss':
            self.vector_store = self._get_faiss_instance(persist_directory)
        else:
            raise ValueError(f"Unsupported vector store: {vector_store}")

        # Initialize the language model (OpenAI for QA)
        self.llm = OpenAI(temperature=0)

        # Set up the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever())

    def _get_chroma_instance(self, persist_directory):
        """Initialize Chroma vector store."""
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        return Chroma(persist_directory=persist_directory, embedding_function=self.embedding_model)

    def _get_faiss_instance(self, persist_directory):
        """Initialize FAISS vector store."""
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        return FAISS.load_local(persist_directory, self.embedding_model) if os.path.exists(persist_directory) else FAISS.from_documents([], self.embedding_model)

    def add_documents(self, documents: list, metadatas: list = None):
        """Add documents to the vector store and persist accordingly."""
        # Add documents to the vector store
        self.vector_store.add_documents(documents, metadatas)

        # Persist based on vector store type
        if isinstance(self.vector_store, Chroma):
            self.vector_store.persist()  # Chroma uses persist
        elif isinstance(self.vector_store, FAISS):
            self.vector_store.save_local()  # FAISS uses save_local

    def search(self, query_text: str, top_k: int = 5):
        """Search the vector store."""
        return self.vector_store.similarity_search_with_score(query_text, k=top_k)

    def qa(self, query_text: str):
        """"QA the LLM"""
        return self.qa_chain(query_text)


# Example usage:
# retriever = Retriever(vector_store="chroma", persist_directory="./chroma_data")
# documents = ["This is a test document.", "Another document for testing."]
# retriever.add_documents(documents)
# result = retriever.query("test", top_k=1)
# print(result)
