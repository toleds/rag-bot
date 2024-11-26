import asyncio
import math

from common import vector_utils, llm_utils
from config import config

from fastapi import HTTPException

from langchain_core.documents import Document
from langchain.chains import RetrievalQA

class DocumentRetriever:
    def __init__(self):
        """
        Initialize the Universal Retriever that can work with either Chroma or FAISS.
        : param config: the configuration setup.
        """

        self.qa = None
        self.llm = None
        self.vector_store_retriever = None
        self.vector_store = None
        self.queue = asyncio.Queue()  # Queue to hold documents
        self.worker_task = None
        self.isProcessing = False

        self.embedding_model = llm_utils.get_embedding_model(config.embeddings.embedding_type, config.embeddings.embedding_model)

        # setup vector_store, retriever, llm
        self.get_or_create_collection()

    async def _worker(self):
        """ Worker task that processes batches of documents from the queue. """
        while self.isProcessing:
            # Get the next batch of documents from the queue
            task = await self.queue.get()
            if task is None:
                self.isProcessing = False

            documents, store_documents = task  # Unpack the tuple
            await self._process_document(documents)

            if store_documents:
                self.store_documents()

            self.queue.task_done()

    async def _start_worker(self):
        """Start a worker to process documents from the queue."""
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

        return self.worker_task


    async def add_documents(self, documents: list[Document], store_documents: bool = False):
        """ Add documents to the queue for processing. """
        self.isProcessing = True
        await self.queue.put((documents,store_documents))
        print(f"Queued  {len(documents)} chunks to the queue. Total documents are {self.queue.qsize()}")

        # Start the worker
        if not self.worker_task:
            asyncio.create_task(self._start_worker())  # Start worker in the background

    async def _process_document(self, documents: list[Document]):
        """
        Add documents to the vector store and persist accordingly.

        :param documents:
        :return:
        """
        # generate ids (PDF only)
        last_page_id = None
        page_index = 0
        stripped_path = f"{config.vector_store.resource_path}/"

        for doc in documents:
            source:str = doc.metadata.get("source").replace(stripped_path, "", 1)

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

            doc.metadata["source"] = source
            doc.id = f"{page_id}:{page_index}"
            doc.metadata["id"] = doc.id

        # Add documents to the vector store
        await self._add_documents_in_batches(documents)


    async def _add_documents_in_batches(self, documents, batch_size=1000):
        print(f"Total chunks to add: {len(documents)}")
        # Split the documents into batches
        num_batches = math.ceil(len(documents) / batch_size)
        print(f"Total batches to add: {num_batches}")
        tasks = []

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch = documents[start:end]

            try:
                # Add the batch of documents asynchronously
                print(f"Adding batch {i + 1}/{num_batches} with {len(batch)} chunks on collection {self.vector_store._collection_name}.")
                tasks.append(self.vector_store_retriever.aadd_documents(batch))
                print(f"Batch {i + 1}/{num_batches} with {len(batch)} chunks added.")
            except Exception as e:
                print(f"Exception: {e}")

        await asyncio.gather(*tasks)
        print("Document batches added.")

    async def search(self, query_text: str):
        """
        Search the vector store.

        :param query_text:
        :return:
        """

        results = await self.vector_store_retriever.ainvoke(query_text)

        if not results:
            raise HTTPException(status_code=404, detail="No similar documents found.  Kindly refine your query.")

        print(f"Results from aget_relevant_documents: {results}")
        print(f"Search Type : {self.vector_store_retriever.search_type} of {self.vector_store_retriever.allowed_search_types}")

        return results, self.vector_store._collection_name

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

        results = await self.vector_store.asimilarity_search_with_score(query_text)

        # If the response is empty, raise 404
        if not results:
            raise HTTPException(status_code=404, detail="No similar documents found.  Kindly refine your query.")

        return results

    async def question_answer(self, query_text: str, documents):
        """
        QA the LLM

        :param documents:
        :param query_text:
        :return:
        """

        print("Sending to LLM to answer...")
        return await self.qa._acall({"query": query_text}), self.vector_store._collection_name

    def store_documents(self):
        """
        Persist based on vector store type

        :return:
        """
        try:
            if config.vector_store.vector_type == 'chroma':
                # self.vector_store.persist()  # Chroma uses persist
                pass
            elif config.vector_store.vector_type == 'faiss':
                self.vector_store.save_local(config.vector_store.data_path)  # FAISS uses save_local
                # Load the vector after saving
                self.vector_store.load_local(
                    folder_path=config.vector_store.data_path,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True)

            print("Document successfully stored.")
        except Exception as e:
            # Catching any exceptions to print a debug message
            print(f"An error occurred during storing documents: {e}")


    def get_or_create_collection(self, collection_name: str = "default"):
        # get the vector store instance
        self.vector_store = vector_utils.get_vector_store(vector_type=config.vector_store.vector_type,
                                                          data_path=config.vector_store.data_path,
                                                          dimension=config.embeddings.dimension,
                                                          embedding_model=self.embedding_model,
                                                          collection_name=collection_name)

        self.vector_store_retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # get te collection (default is "default")
        print(f"Current collection in use {self.vector_store._collection_name}")

        # Initialize the language model (OpenAI for QA)
        self.llm = llm_utils.get_llm(llm_type=config.llms.llm_type,
                                     model_name=config.llms.llm_name,
                                     local_server=config.llms.local_server)

        # Set up the RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(llm=self.llm,
                                              chain_type="stuff",
                                              retriever=self.vector_store_retriever,
                                              verbose=True,
                                              return_source_documents=True)

        return  self.vector_store._collection_name

    def get_collection_list(self):
        return self.vector_store._client.list_collections()


    def initialize_vector_store(self):
        self.vector_store.reset_collection()
        self.vector_store = vector_utils.get_vector_store(vector_type=config.vector_store.vector_type,
                                                   data_path=config.vector_store.data_path,
                                                   embedding_model=self.embedding_model,
                                                   dimension=config.embeddings.dimension)


    # def filter_unique_documents(self, documents):
    #     ids = self.vector_store.get(where={"source":documents[0].metadata["source"]}, include=[])
    #     print(f"Doc Ids : {ids}")
    #     for doc in documents:
    #         print(f"Check Id: {doc.id}")
    #         if doc.metadata["id"] in ids:
    #             print(f"Duplicate Id: {doc.id}")
    #         else:
    #             print(f"Unique id: {doc.id}")




