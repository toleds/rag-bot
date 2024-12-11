import math
import asyncio

from common import vector_utils, llm_utils
from config import config
from langchain_core.documents import Document


class DocumentRetriever:
    def __init__(self):
        """
        Initialize the Universal Retriever that can work with either Chroma or FAISS.
        : param config: the configuration setup.
        """

        self.vector_store_retriever = None
        self.vector_store = None

        self.worker_task = None
        self.isProcessing = False
        self.queue = asyncio.Queue()  # Queue to hold documents

        self.embedding_model = llm_utils.get_embedding_model(
            config.embeddings.embedding_type, config.embeddings.embedding_model
        )

        # setup vector_store, retriever, llm
        self.get_or_create_collection()

    async def _worker(self):
        """Worker task that processes batches of documents from the queue."""
        while self.isProcessing:
            # Get the next batch of documents from the queue
            task = await self.queue.get()
            if task is None:
                self.isProcessing = False

            documents = task  # Unpack the tuple
            await self._process_document(documents)

            self.queue.task_done()

    async def _start_worker(self):
        """Start a worker to process documents from the queue."""
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

        return self.worker_task

    async def add_documents(self, documents: list[Document]):
        """Add documents to the queue for processing."""
        self.isProcessing = True
        print(f"Adding to queue: {len(documents)} documents.")
        await self.queue.put(documents)
        print(
            f"Queued  {len(documents)} chunks to the queue. Total documents are {self.queue.qsize()}"
        )

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
            source: str = doc.metadata.get("source").replace(stripped_path, "", 1)

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

    async def _add_documents_in_batches(self, documents, batch_size=10):
        print(f"Total chunks to add: {len(documents)}")
        # Split the documents into batches
        num_batches = math.ceil(len(documents) / batch_size)
        print(f"Total batches to add: {num_batches}")

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch = documents[start:end]

            try:
                # Add the batch of documents asynchronously
                print(
                    f"Adding batch {i + 1}/{num_batches} with {len(batch)} chunks on collection {self.vector_store._collection_name}."
                )

                await self.vector_store.aadd_documents(batch)
                print(f"Batch {i + 1}/{num_batches} with {len(batch)} chunks added.")
            except Exception as e:
                print(f"Exception: {e}")

        print("Document batches added.")

    def retrieve(self, query: str):
        """
        Search the vector store.

        :return:
        """

        documents = self.vector_store_retriever.invoke(query)

        return documents

    def get_or_create_collection(self, collection_name: str = "default"):
        # get the vector store instance
        self.vector_store = vector_utils.get_vector_store(
            data_path=config.vector_store.data_path,
            embedding_model=self.embedding_model,
            collection_name=collection_name,
        )

        self.vector_store_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )

        return self.vector_store._collection_name

    def get_collection_list(self):
        return self.vector_store._client.list_collections()

    def initialize_vector_store(self):
        self.vector_store.reset_collection()
        self.vector_store = vector_utils.get_vector_store(
            data_path=config.vector_store.data_path,
            embedding_model=self.embedding_model,
        )
