import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ValidationError

load_dotenv()

class VectorStoreConfig(BaseSettings):
    vector_type: str
    data_path: str
    resource_path: str

class LlmConfig(BaseSettings):
    llm_type: str
    temperature: float
    llm_name: str
    api_key: str
    local_server: str

class Embeddings(BaseSettings):
    embedding_model: str
    embedding_type: str
    dimension: int

class AppConfig(BaseSettings):
    vector_store: VectorStoreConfig
    llms: LlmConfig
    embeddings: Embeddings

    class Config:
        env_file = "/.env"  # Specify the path to your .env file
        env_nested_delimiter = "__"  # Allow nested environment variable syntax

try:
    config = AppConfig()

    # Ensure the directory exists
    if not os.path.exists(config.vector_store.data_path):
        os.makedirs(config.vector_store.data_path)
        print(f"Created Data directory: {config.vector_store.data_path}")
    else:
        print(f"Using existing Data directory: {config.vector_store.data_path}")

    if not os.path.exists(config.vector_store.resource_path):
        os.makedirs(config.vector_store.resource_path)
        print(f"Created Resource directory: {config.vector_store.resource_path}")
    else:
        print(f"Using existing Resource directory: {config.vector_store.resource_path}")

    # set openapi key
    os.environ["OPENAI_API_KEY"] = config.llms.api_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.llms.api_key
except ValidationError as err:
    raise RuntimeError(f"Invalid configuration: {err}") from err