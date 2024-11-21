from common import utils, vector_utils, llm_utils
from config import config

from fastapi import HTTPException

from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

class DocumentRetriever:
    def __init__(self):
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
        self.embedding_model = llm_utils.get_embedding_model(config.embeddings.embedding_type, config.embeddings.embedding_model)

        # get the vector store instance
        self.vector_store = vector_utils.get_vector_store(vector_store_type=config.vector_store.vector_type,
                                                   data_path=config.vector_store.data_path,
                                                   dimension=config.embeddings.dimension,
                                                   embedding_model=self.embedding_model)

        # Initialize the language model (OpenAI for QA)
        self.llm = llm_utils.get_llm(llm_type=config.llms.llm_type,
                                 model_name=config.llms.llm_name,
                                 local_server=config.llms.local_server)

        # Set up the RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(llm=self.llm,
                                              chain_type="stuff",
                                              retriever=self.vector_store.as_retriever(),
                                              verbose=True,
                                              return_source_documents=True)

    async def add_documents(self, documents: list[Document], store_documents: bool = False):
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
        await self.vector_store.aadd_documents(documents)

        if store_documents:
            self.store_documents()

        print("Documents stored successfully.")

    async def search(self, query_text: str):
        """
        Search the vector store.

        :param query_text:
        :return:
        """

        return await self.vector_store.asimilarity_search(query_text, k=5)

    async def search_with_score(self, query_text: str, filter_score: float = 1.0):
        """
        Search the vector store.

        :param filter_score:
        :param query_text:
        :return:
        """
        results = await self.vector_store.asimilarity_search_with_score(query_text, k=5)
        filtered_results = [doc for doc, score in results if score < filter_score]

        # If the response is empty, raise 404
        if not filtered_results:
            raise HTTPException(status_code=404, detail="No similar documents found.  Kindly refine your query.")

        return filtered_results

    async def search_with_score_no_fiter(self, query_text: str):
        """
        Search the vector store.

        :param query_text:
        :return:
        """

        results = await self.vector_store.asimilarity_search_with_score(query_text, k=5)

        # If the response is empty, raise 404
        if not results:
            raise HTTPException(status_code=404, detail="No similar documents found.  Kindly refine your query.")

        return results

    async def question_answer(self, query_text: str, documents: list):
        """
        QA the LLM

        :param documents:
        :param query_text:
        :return:
        """
        context = utils.format_context(documents)
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        query = prompt_template.format(context=context, query=query_text)

        return await self.qa._acall({"query": query})  #invoke(query)

    def store_documents(self):
        """
        Persist based on vector store type

        :return:
        """
        if config.vector_store.vector_store_type == 'chroma':
            # self.vector_store.persist()  # Chroma uses persist
            pass
        elif config.vector_store.vector_store_type == 'faiss':
            self.vector_store.save_local(config.vector_store.data_path)  # FAISS uses save_local

    def initialize_vector_store(self):
        self.vector_store = vector_utils.get_vector_store(vector_store_type=config.vector_store.vector_type,
                                                   data_path=config.vector_store.data_path,
                                                   embedding_model=self.embedding_model,
                                                   dimension=config.embeddings.dimension,
                                                   initialize=True)


    # def _get_existing_doc_ids(self):
    #     if config.vector_store.vector_type == 'chroma':
    #         self.vector_store.get()
    #         pass
    #     elif config.vector_store.vector_type == 'faiss':
    #         self.vector_store.get
    #         pass
    #     else:
    #         raise ValueError(f"Unsupported vector store: {config.vector_store.vector_type}")