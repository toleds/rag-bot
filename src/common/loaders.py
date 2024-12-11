import os
import re

from bs4 import BeautifulSoup
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    RecursiveUrlLoader,
)
from langchain_core.documents import Document


chunk_size = 1000
overlap = 200

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=overlap,
    length_function=len,
    is_separator_regex=False,
)


async def load_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts text from a PDF file, splits the text from each page into chunks, and returns a list of text chunks.

    :param pdf_path: Path to the PDF file.
    :return: List of text chunks extracted from the PDF.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Load the file
    loader = PyPDFLoader(pdf_path)
    documents = await loader.aload()

    # Split the text into chunks
    text_chunks = text_splitter.split_documents(documents)
    print(
        f"Extracted and split text into {len(text_chunks)} chunks from PDF: {pdf_path}"
    )

    return text_chunks


async def load_text_file(file_path: str) -> List[Document]:
    """
    Extracts text from a plain text file, splits it into chunks, and returns a list of text chunks.

    :param file_path: Path to the text file.
    :return: List of text chunks extracted from the text file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    # Load the file
    loader = TextLoader(file_path)
    documents = await loader.aload()

    # Split the text into chunks
    text_chunks = text_splitter.split_documents(documents)
    print(
        f"Extracted and split text into {len(text_chunks)} chunks from file: {file_path}"
    )

    return text_chunks


async def load_web_url(root_url: str) -> List[Document]:
    """
    Extracts text from a web urls text file, splits it into chunks, and returns a list of text chunks.

    :param root_url:
    :return:
    """

    def bs4_extractor(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

    print("Extracting list of web pages....")
    pages = RecursiveUrlLoader(
        url=root_url, max_depth=5, prevent_outside=True, extractor=bs4_extractor
    )

    print("Collecting documents from web pages.....")
    documents = await pages.aload()

    # for doc in pages.alazy_load():
    #     documents.append(doc)

    # Split the text into chunks
    text_chunks = text_splitter.split_documents(documents)
    print(
        f"Extracted and split text into {len(text_chunks)} chunks from urls: {root_url}"
    )

    return text_chunks
