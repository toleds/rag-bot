import os

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_openai import OpenAIEmbeddings, OpenAI


def extract_text_from_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Extracts text from a PDF file, splits the text from each page into chunks, and returns a list of text chunks.

    :param pdf_path: Path to the PDF file.
    :param chunk_size: Maximum size of each text chunk.
    :param chunk_overlap: Overlap size between chunks.
    :return: List of text chunks extracted from the PDF.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Load the file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split the text into chunks
    text_chunks = text_splitter.split_documents(documents)
    print(f"Extracted and split text into {len(text_chunks)} chunks from PDF: {pdf_path}")

    return text_chunks

def extract_text_from_file(file_path: str, chunk_size: int = 500, chunk_overlap: int = 0) -> List[Document]:
    """
    Extracts text from a plain text file, splits it into chunks, and returns a list of text chunks.

    :param file_path: Path to the text file.
    :param chunk_size: Maximum size of each text chunk.
    :param chunk_overlap: Overlap size between chunks.
    :return: List of text chunks extracted from the text file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    # Load the file
    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)

    # Split the text into chunks
    text_chunks = text_splitter.split_documents(documents)
    print(f"Extracted and split text into {len(text_chunks)} chunks from file: {file_path}")

    return text_chunks

def get_vector_store(vector_store_type: str, persist_directory, embedding_model):
    """
    decide which vector store to use

    :param vector_store_type:
    :param persist_directory:
    :param embedding_model:
    :return:
    """
    # Decide which vector store to use (Chroma or FAISS)
    if vector_store_type == 'chroma':
        return get_chroma_instance(persist_directory=persist_directory, embedding_model=embedding_model)
    elif vector_store_type == 'faiss':
        return get_faiss_instance(persist_directory=persist_directory, embedding_model=embedding_model)
    else:
        raise ValueError(f"Unsupported vector store: {vector_store_type}")

def get_embedding_model(embedding_model: str = "openai"):
    """
    decide which embedding model to use

    :param embedding_model:
    :return:
    """
    # Decide which embedding model to use
    if embedding_model == "openai":
        return get_openai_embedding()
    elif embedding_model == "huggingface":
        return get_huggingface_embedding()
    else:
        raise ValueError(f"Unsupported embedding: {embedding_model}")

def get_llm(llm: str, temperature: float = 0.5):
    """
    decide which LLM to use

    :param llm:
    :param temperature:
    :return:
    """
    # Decide which llm to use
    if llm == "openai":
        return get_openai_llm(temperature)
    elif llm == "huggingface":
        return get_hugging_face_llm(temperature)
    else:
        raise ValueError(f"Unsupported llm: {llm}")

def get_chroma_instance(persist_directory, embedding_model):
    """
    Initialize Chroma vector store.

    :param persist_directory:
    :param embedding_model:
    :return:
    """
    _verify_or_create_vector_store_folder(persist_directory)

    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def get_faiss_instance(persist_directory, embedding_model):
    """
    Initialize FAISS vector store.

    :param persist_directory:
    :param embedding_model:
    :return:
    """
    _verify_or_create_vector_store_folder(persist_directory)

    # Check if the necessary files exist
    if not os.path.exists(os.path.join(persist_directory, "index.faiss")) or not os.path.exists(os.path.join(persist_directory, "index.pkl")):
        print("Files not found. Initializing a new FAISS index...")
        # Initialize FAISS and save it
        index = IndexFlatL2(len(embedding_model.embed_query("dummy")))
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={})

        vector_store.save_local(persist_directory)
        print("FAISS index and docstore have been saved.")
    else:
        # Load existing vector store
        vector_store = FAISS.load_local(folder_path=persist_directory,
                                        embeddings=embedding_model,
                                        allow_dangerous_deserialization=True)
        print("Loaded existing FAISS vector store.")

    return vector_store

def get_openai_embedding():
    """
    OpenAI embedding model

    :return:
    """
    return OpenAIEmbeddings()  # default embedding model

def get_huggingface_embedding():
    """
    HuggingFace embedding model

    :return:
    """
    return HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

def get_openai_llm(temperature: float = 0.5):
    """

    :param temperature:
    :return:
    """
    return OpenAI(temperature=temperature)

def get_hugging_face_llm(temperature: float = 0.5):
    """
    HuggingFace LLM

    :param temperature:
    :return:
    """
    return HuggingFaceEndpoint(repo_id = "gpt2", temperature=temperature, max_new_tokens=100)

def _verify_or_create_vector_store_folder(persist_directory):
    """
    Create or verify folder exists

    :param persist_directory:
    :return:
    """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

def format_context(documents):
    """
    reformat document to String

    :param documents:
    :return:
    """
    context = ""
    for i, doc in enumerate(documents):
        context += doc.page_content + "\n"
    return context
