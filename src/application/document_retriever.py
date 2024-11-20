from fastapi import HTTPException
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

from common import utils
from config import AppConfig


class DocumentRetriever:
    def __init__(self, config: AppConfig):
        """
        Initialize the Universal Retriever that can work with either Chroma or FAISS.
        : param config: the configuration setup.
        """

        self.PROMPT_TEMPLATE = """
        Answer the query based only on the following context:
        {context}

        ---
        Answer the query based on the above context: {query}
        
        Use markdown formatting on the response.
        """
        self.config = config
        self.embedding_model = utils.get_embedding_model(config.embeddings.embedding_type, config.embeddings.embedding_model)
        self.vector_store_type = config.vector_store.vector_type
        self.data_path = config.vector_store.data_path

        # get the vector store instance
        self.vector_store = utils.get_vector_store(vector_store_type=config.vector_store.vector_type,
                                                   data_path=config.vector_store.data_path,
                                                   dimension=config.embeddings.dimension,
                                                   embedding_model=self.embedding_model)

        # Initialize the language model (OpenAI for QA)
        self.llm = utils.get_llm(llm_type=config.llms.llm_type,
                                 model_name=config.llms.llm_name,
                                 local_server=self.config.llms.local_server)

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
        # generate ids (PDF only)
        last_page_id = None
        page_index = 0

        for doc in documents:
            source:str = doc.metadata.get("source")

            if source.lower().endswith(".pdf"):
                page = doc.metadata.get("page")
                page_id = f"{source}:{page}"
            else:
                page_id = f"{source}"

            if page_id == last_page_id:
                page_index += 1
            else:
                page_index = 0

            last_page_id = page_id
            doc.metadata["id"] = f"{page_id}:{page_index}"

        # Add documents to the vector store
        self.vector_store.add_documents(documents)

        if store_documents:
            self.store_documents()

        print("Documents stored successfully.")

    def search(self, query_text: str):
        """
        Search the vector store.

        :param query_text:
        :return:
        """

        return self.vector_store.similarity_search(query_text, k=5)

    def search_with_score(self, query_text: str, filter_score: float = 1.0):
        """
        Search the vector store.

        :param filter_score:
        :param query_text:
        :return:
        """
        results = self.vector_store.similarity_search_with_score(query_text, k=5)
        filtered_results = [doc for doc, score in results if score < filter_score]

        # If the response is empty, raise 404
        if not filtered_results:
            raise HTTPException(status_code=404, detail="No similar documents found.  Kindly refine your query.")

        return filtered_results

    def search_with_score_no_fiter(self, query_text: str):
        """
        Search the vector store.

        :param query_text:
        :return:
        """

        results = self.vector_store.similarity_search_with_score(query_text, k=5)

        # If the response is empty, raise 404
        if not results:
            raise HTTPException(status_code=404, detail="No similar documents found.  Kindly refine your query.")

        return results

    def question_answer(self, query_text: str, documents: list):
        """
        QA the LLM

        :param documents:
        :param query_text:
        :return:
        """
        context = utils.format_context(documents)
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        query = prompt_template.format(context=context, query=query_text)

        return self.qa.invoke(query)

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

    def initialize_vector_store(self):
        self.vector_store = utils.get_vector_store(vector_store_type=self.config.vector_store.vector_type,
                                                   data_path=self.config.vector_store.data_path,
                                                   embedding_model=self.embedding_model,
                                                   dimension=self.config.embeddings.dimension,
                                                   initialize=True)

