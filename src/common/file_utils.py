import os

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document


def extract_text_from_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len,
                                                   is_separator_regex=False)

    # Split the text into chunks
    text_chunks = text_splitter.split_documents(documents)
    print(f"Extracted and split text into {len(text_chunks)} chunks from PDF: {pdf_path}")

    return text_chunks

def extract_text_from_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len,
                                                   is_separator_regex=False)

    # Split the text into chunks
    text_chunks = text_splitter.split_documents(documents)
    print(f"Extracted and split text into {len(text_chunks)} chunks from file: {file_path}")

    return text_chunks