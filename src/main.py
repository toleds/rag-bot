import yaml, os
from .wrapper.model_wrapper import OpenApiModel
from .wrapper.rag_wrapper import RAGWrapper
from .retriever.chromadb_retriever import ChromaDBRetriever
from .retriever.faiss_retriever import FAISSRetriever

# Helper function to load configuration from a YAML file
def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Factory function to get the document retriever based on the configuration
def get_document_retriever(retriever_type: str, persist_path: str = "./vector_data"):
    # Ensure the directory exists
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)
        print(f"Created directory: {persist_path}")
    else:
        print(f"Using existing directory: {persist_path}")

    if retriever_type == "chroma":
        return ChromaDBRetriever(persist_directory=persist_path)
    elif retriever_type == "faiss":
        return FAISSRetriever()
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")

# Main function
def main():
    # Load configuration
    config = load_config("config.yaml")

    # Initialize the model wrapper (e.g., OpenAI, Cohere)
    model_type = config["model"]["type"]
    api_key = config["model"]["api_key"]
    model = OpenApiModel(model_type=model_type, api_key=api_key)

    # Initialize the document retriever
    retriever_type = config["retriever"]["type"]
    persist_path = config.get("retriever", {}).get("persist_path", "./vector_data")
    retriever = get_document_retriever(retriever_type=retriever_type, persist_path=persist_path)

    # Initialize the RAG wrapper
    rag_wrapper = RAGWrapper(model=model, retriever=retriever)

    # Sample data to add to the vector store
    documents = [
        "The bank offers a variety of credit cards with different rewards programs.",
        "Our savings accounts provide competitive interest rates.",
        "The bank's mortgage plans include fixed and variable interest rate options."
    ]
    metadatas = [
        {"source": "bank_product_guide"},
        {"source": "bank_website"},
        {"source": "mortgage_brochure"}
    ]

    # Add documents to the retriever (vector store)
    print("Adding documents to the vector store...")
    # retriever.add_documents(documents, metadatas)

    # Query the retriever and generate responses using RAG
    query_text = "quantum physics?"
    print(f"Querying with: {query_text}")
    response = rag_wrapper.generate_response(query_text)

    # Print the response
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()

