import yaml, os

from config import AppConfig
from question_answer import QuestionAnswer
from document_retriever import DocumentRetriever
from langchain_core.documents import Document

# Helper function to load configuration from a YAML file
def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Main function
def main():
    # Load configuration
    config = AppConfig.from_yaml("config.yaml")

    # set openapi key
    os.environ["OPENAI_API_KEY"] = config.openapi.api_key

    # Initialize the document retriever
    retriever = DocumentRetriever(vector_store=config.vector_store.vector_type, persist_directory=config.vector_store.persist_directory)

    # Initialize the QA Wrapper
    question_answer = QuestionAnswer(retriever=retriever)

    # ===================================================================================
    # Sample data to add to the vector store
    page_content = [
        "The bank offers a variety of credit cards with different rewards programs.",
        "Our savings accounts provide competitive interest rates.",
        "The bank's mortgage plans include fixed and variable interest rate options."
    ]
    metadata = [
        {"source": "bank_product_guide"},
        {"source": "bank_website"},
        {"source": "mortgage_brochure"}
    ]

    # Convert to a list of Document objects
    documents = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(page_content, metadata)
    ]
    # ===================================================================================

    # Add documents to the retriever (vector store)
    print("Adding documents to the vector store...")
    retriever.add_documents(documents)

    # Query the retriever and generate responses using RAG
    query_text = "quantum physics?"

    # get similarities
    response_similarities = question_answer.generate_similarities(query_text)
    print(f"Similarities: {response_similarities}")

    # get answer
    response_answer = question_answer.generate_response(query_text)
    print(f"Question: {query_text}")
    print(f"Answer: {response_answer}")


if __name__ == "__main__":
    main()

