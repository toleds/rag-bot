import yaml


class VectorStoreConfig:
    def __init__(self, vector_type: str, data_path: str, resource_path: str):
        self.vector_type = vector_type
        self.data_path = data_path
        self.resource_path = resource_path

class LlmConfig:
    def __init__(self, llm: str, openapi_key: str, hugging_face_key: str, temperature: float):
        self.llm = llm
        self.openapi_key = openapi_key
        self.hugging_face_key = hugging_face_key
        self.temperature = temperature

class Embeddings:
    def __init__(self, embedding_model: str):
        self.embedding_model =  embedding_model

class AppConfig:
    def __init__(self, vector_store: dict, llms: dict, embeddings: dict):
        self.vector_store = VectorStoreConfig(**vector_store)
        self.llms = LlmConfig(**llms)
        self.embeddings = Embeddings(**embeddings)

    @staticmethod
    def from_yaml(file_path: str) -> "AppConfig":
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return AppConfig(**config)
