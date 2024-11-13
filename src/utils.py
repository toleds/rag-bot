from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document

import PyPDF2, os

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

    text_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Extract text and split per page
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                page_text = text.strip()
                # Split the text from this page into chunks
                page_chunks = text_splitter.split_documents(page_text)
                print(f"Split page {page_num + 1} into {len(page_chunks)} chunks.")
                text_chunks.extend(page_chunks)

    print(f"Extracted and split text into {len(text_chunks)} chunks from PDF: {pdf_path}")
    return text_chunks

def extract_text_from_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Extracts text from a plain text file, splits it into chunks, and returns a list of text chunks.

    :param file_path: Path to the text file.
    :param chunk_size: Maximum size of each text chunk.
    :param chunk_overlap: Overlap size between chunks.
    :return: List of text chunks extracted from the text file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    text_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Read the content of the text file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().strip()

    # Split the text into chunks
    text_chunks = text_splitter.split_documents(text)
    print(f"Extracted and split text into {len(text_chunks)} chunks from file: {file_path}")

    return text_chunks

def get_chroma_instance(persist_directory, embedding_model):
    """Initialize Chroma vector store."""
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def get_faiss_instance(persist_directory, embedding_model):
    """Initialize FAISS vector store."""
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    return FAISS.load_local(persist_directory, embedding_model) if os.path.exists(persist_directory) else FAISS.from_documents([], embedding_model)
