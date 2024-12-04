import os
import shutil

from langchain_chroma import Chroma


def get_vector_store(
    data_path,
    embedding_model,
    collection_name: str = "default",
):
    """
    decide which vector store to use

    :param collection_name:
    :param data_path:
    :param embedding_model:
    :return:
    """

    return get_chroma_instance(
        data_path=data_path,
        embedding_model=embedding_model,
        collection_name=collection_name,
    )


def get_chroma_instance(data_path, embedding_model, collection_name: str = "default"):
    """
    Initialize Chroma vector store.

    :param collection_name:
    :param data_path:
    :param embedding_model:
    :return:
    """
    return Chroma(
        persist_directory=data_path,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )


def clear_database(path: str):
    """
    init faiss dbb
    :param path:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Data path {path} deleted!")
