import os

from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

def get_vector_store(vector_store_type: str, data_path, embedding_model, dimension: int, initialize: bool = False):
    """
    decide which vector store to use

    :param dimension:
    :param initialize:
    :param vector_store_type:
    :param data_path:
    :param embedding_model:
    :return:
    """
    # Decide which vector store to use (Chroma or FAISS)
    if vector_store_type == 'chroma':
        return get_chroma_instance(data_path=data_path, embedding_model=embedding_model)
    elif vector_store_type == 'faiss':
        return get_faiss_instance(data_path=data_path, embedding_model=embedding_model, initialize=initialize, dimension=dimension)
    else:
        raise ValueError(f"Unsupported vector store: {vector_store_type}")

def get_chroma_instance(data_path, embedding_model):
    """
    Initialize Chroma vector store.

    :param data_path:
    :param embedding_model:
    :return:
    """
    return Chroma(persist_directory=data_path, embedding_function=embedding_model)

def _init_faiss_instance(embedding_model, dimension: int):
    """
    init faiss dbb
    :param embedding_model:
    :return:
    """
    # Initialize FAISS and save it
    index = IndexFlatL2(dimension)
    # always reset
    index.reset()

    return FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={})

def get_faiss_instance(data_path, embedding_model, initialize, dimension: int):
    """
    Initialize FAISS vector store.

    :param dimension:
    :param initialize:
    :param data_path:
    :param embedding_model:
    :return:
    """

    # Check if the necessary files exist
    if (not os.path.exists(os.path.join(data_path, "index.faiss"))
            or not os.path.exists(os.path.join(data_path, "index.pkl"))
            or initialize):
        print("Initializing a new FAISS index...")
        # Initialize FAISS and save it
        vector_store = _init_faiss_instance(embedding_model, dimension)
        vector_store.save_local(data_path)
        print("FAISS index and doc store have been initialized.")
    else:
        # Load existing vector store
        vector_store = FAISS.load_local(folder_path=data_path,
                                        embeddings=embedding_model,
                                        allow_dangerous_deserialization=True)
        print("Loaded existing FAISS vector store.")

    return vector_store
