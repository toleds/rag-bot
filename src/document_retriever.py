
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import os, utils

class DocumentRetriever:
    def __init__(self, vector_store: str, persist_directory: str = "./vector_data"):
        """
        Initialize the Universal Retriever that can work with either Chroma or FAISS.
        :param vector_store: The type of vector store ('chroma' or 'faiss').
        :param persist_directory: Directory to store the vector store data.
        """
        # Ensure the directory exists
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            print(f"Created directory: {persist_directory}")
        else:
            print(f"Using existing directory: {persist_directory}")

        self.embedding_model = OpenAIEmbeddings()
        self.vector_store_type = vector_store

        # get the vector store instance
        self._get_vector_store(persist_directory)

        # Initialize the language model (OpenAI for QA)
        self.llm = OpenAI(temperature=0)

        # Set up the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever())

    def add_documents(self, documents: list, store_documents: bool = False):
        """Add documents to the vector store and persist accordingly."""
        # Add documents to the vector store
        self.vector_store.add_documents(documents)

        if store_documents:
            self.store_documents()

    def search(self, query_text: str, top_k: int = 5):
        """Search the vector store."""
        return self.vector_store.similarity_search(query_text, k=top_k)

    def search_with_score(self, query_text: str, top_k: int = 5):
        """Search the vector store."""
        return self.vector_store.similarity_search_with_score(query_text, k=top_k)

    def question_answer(self, query_text: str):
        """"QA the LLM"""
        return self.qa_chain(query_text)

    def store_documents(self):
        # Persist based on vector store type
        if self.vector_store_type == 'chroma':
            self.vector_store.persist()  # Chroma uses persist
        elif self.vector_store_type == 'faiss':
            self.vector_store.save_local()  # FAISS uses save_local

    def _get_vector_store(self, persist_directory: str):
        # Decide which vector store to use (Chroma or FAISS)
        if self.vector_store_type == 'chroma':
            self.vector_store = utils.get_chroma_instance(persist_directory=persist_directory, embedding_model=self.embedding_model)
        elif self.vector_store_type == 'faiss':
            self.vector_store = utils.get_faiss_instance(persist_directory=persist_directory, embedding_model=self.embedding_model)
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_store_type}")
