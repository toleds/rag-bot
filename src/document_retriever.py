from fastapi import HTTPException
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

import utils
from src.config import AppConfig


class DocumentRetriever:
    def __init__(self, config: AppConfig):
        """
        Initialize the Universal Retriever that can work with either Chroma or FAISS.
        : param config: the configuration setup.
        """
        self.embedding_type = utils.get_embedding_model(config.embeddings.embedding_type, config.embeddings.embedding_model)
        self.vector_store_type = config.vector_store.vector_type
        self.data_path = config.vector_store.data_path

        # get the vector store instance
        self.vector_store = utils.get_vector_store(vector_store_type=config.vector_store.vector_type,
                                                   data_path=config.vector_store.data_path,
                                                   embedding_model=self.embedding_type)

        # Initialize the language model (OpenAI for QA)
        self.llm = utils.get_llm(llm_type=config.llms.llm_type,
                                 model_name=config.llms.model_name,
                                 task=config.llms.task,
                                 temperature=config.llms.temperature)

        # Set up the RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(llm=self.llm,
                                              chain_type="stuff",
                                              retriever=self.vector_store.as_retriever(),
                                              verbose=True,
                                              return_source_documents=True)

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
        return self.vector_store.similarity_search(query_text, k=2, fetch_k=top_k)

    def search_with_score(self, query_text: str, top_k: int = 3, filter_score: float = 0.7):
        """
        Search the vector store.

        :param filter_score:
        :param query_text:
        :param top_k:
        :return:
        """

        results = self.vector_store.similarity_search_with_score(query_text, top_k)
        filtered_results = [doc for doc, score in results if score < filter_score]

        # If the response is empty, raise 404
        if not filtered_results:
            raise HTTPException(status_code=404, detail="No similar documents found.  Kindly refine your query.")

        return filtered_results

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
            self.vector_store.save_local(self.data_path)  # FAISS uses save_local

