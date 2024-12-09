from langchain_ollama import ChatOllama
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


def get_embedding_model(
    embedding_type: str = "openai",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    decide which embedding model to use

    :param embedding_model:
    :param embedding_type:
    :return:
    """
    # Decide which embedding model to use
    if embedding_type == "openai":
        return get_openai_embedding(embedding_model=embedding_model)
    elif embedding_type == "huggingface":
        return get_huggingface_embedding(embedding_model=embedding_model)
    else:
        raise ValueError(f"Unsupported embedding: {embedding_model}")


def get_llm(llm_type: str, model_name: str, local_server: str):
    """
    decide which LLM to use

    :param local_server:
    :param model_name:
    :param llm_type:
    :return:
    """
    # Decide which llm to use
    if llm_type == "openai":
        return get_openai_llm(model_name=model_name)
    elif llm_type == "huggingface":
        return get_hugging_face_llm(model_name=model_name)
    elif llm_type == "ollama":
        return get_ollama_llm(model_name=model_name, local_server=local_server)
    else:
        raise ValueError(f"Unsupported llm: {llm_type}")


def get_openai_embedding(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    OpenAI embedding model

    :return:
    """
    return OpenAIEmbeddings(model=embedding_model)  # default embedding model


def get_huggingface_embedding(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    HuggingFace embedding model

    :return:
    """
    return HuggingFaceEmbeddings(model_name=embedding_model)


def get_openai_llm(model_name: str):
    """

    :param model_name:
    :return:
    """
    return ChatOpenAI(model_name=model_name)


def get_hugging_face_llm(model_name: str):
    """
    HuggingFace LLM

    :param model_name:
    :return:
    """
    llm = HuggingFaceEndpoint(repo_id=model_name)

    return ChatHuggingFace(llm=llm)


def get_ollama_llm(model_name: str, local_server: str):
    """

    :param local_server:
    :param model_name:
    :return:
    """

    return ChatOllama(model=model_name, base_url=local_server)
