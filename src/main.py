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
    document_retriever = DocumentRetriever(vector_store_type=config.vector_store.vector_type,
                                  embedding_model=config.embeddings.embedding_model,
                                  persist_directory=config.vector_store.persist_directory)

    # Initialize the QA Wrapper
    question_answer = QuestionAnswer(retriever=document_retriever)

    # ===================================================================================
    # Sample data to add to the vector store (list, pdf, txt)
    page_content = [
        "The bank offers a variety of credit cards with different rewards programs.",
        "Our savings accounts provide competitive interest rates.",
        "The bank's mortgage plans include fixed and variable interest rate options.",
        "Philippines is located in south east asia",
        "The capital of the Philippines is Manila",
        "chowpin is bald"
    ]
    metadata = [
        {"source": "bank_product_guide"},
        {"source": "bank_website"},
        {"source": "mortgage_brochure"},
        {"source": "wikipedia"},
        {"source": "wikipedia"},
        {"source": "barber_shop"}
    ]

    # Convert to a list of Document objects
    documents = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(page_content, metadata)
    ]

    # document = utils.extract_text_from_pdf(pdf_path=pdf_path)
    # document = utils.extract_text_from_file(file_path=file_path)
    # ===================================================================================

    # Add documents to the retriever (vector store)
    # print("Adding documents to the vector store...")
    # document_retriever.add_documents(documents, True)

    # Query the retriever and generate responses using RAG
    query_text = "chowpin"
    print(f"Question: {query_text}")

    # get similarities
    response_similarities = question_answer.generate_similarities(query_text)
    print(f"Similarities: {response_similarities}")

    # get similarities
    response_similarities_with_score = question_answer.generate_similarities_with_score(query_text)
    print(f"Similarities with score: {response_similarities_with_score}")

    # get answer from LLM (final format)
    # response_answer = question_answer.generate_response(query_text)
    # print(f"Question: {query_text}")
    # print(f"Answer: {response_answer}")


if __name__ == "__main__":
    main()

