from langchain_core.documents import Document
from langchain.chains import RetrievalQA

import os, utils


from src.config import AppConfig


class DocumentRetriever:
    def __init__(self, config: AppConfig):
        """
        Initialize the Universal Retriever that can work with either Chroma or FAISS.
        : param config: the configuration setup.
        """
        # Ensure the directory exists
        if not os.path.exists(config.vector_store.persist_directory):
            os.makedirs(config.vector_store.persist_directory)
            print(f"Created directory: {config.vector_store.persist_directory}")
        else:
            print(f"Using existing directory: {config.vector_store.persist_directory}")

        self.embedding_model = utils.get_embedding_model(config.embeddings.embedding_model)
        self.vector_store_type = config.vector_store.vector_type
        self.persist_directory = config.vector_store.persist_directory

        # get the vector store instance
        self.vector_store = utils.get_vector_store(vector_store_type=config.vector_store.vector_type,
                                                   persist_directory=config.vector_store.persist_directory,
                                                   embedding_model=self.embedding_model)

        # Initialize the language model (OpenAI for QA)
        self.llm = utils.get_llm(llm=config.llms.llm, temperature=config.llms.temperature)

        # Set up the RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever())

    def add_documents(self, documents: list[Document], store_documents: bool = False):
        """
        Add documents to the vector store and persist accordingly.

        :param documents:
        :param store_documents:
        :return:
        """
        # Add documents to the vector store
        self.vector_store.add_documents(documents)

        if store_documents:
            self.store_documents()

    def search(self, query_text: str, top_k: int = 3):
        """
        Search the vector store.

        :param query_text:
        :param top_k:
        :return:
        """
        return self.vector_store.similarity_search(query_text, top_k)

    def search_with_score(self, query_text: str, top_k: int = 3):
        """
        Search the vector store.

        :param query_text:
        :param top_k:
        :return:
        """
        return self.vector_store.similarity_search_with_score(query_text, top_k)

    def question_answer(self, query_text: str, context: str):
        """
        QA the LLM

        :param context:
        :param query_text:
        :return:
        """
        return self.qa.invoke({"query": query_text, "context": context})

    def store_documents(self):
        """
        Persist based on vector store type

        :return:
        """
        if self.vector_store_type == 'chroma':
            # self.vector_store.persist()  # Chroma uses persist
            pass
        elif self.vector_store_type == 'faiss':
            self.vector_store.save_local(self.persist_directory)  # FAISS uses save_local

