import yaml, os, utils

from config import AppConfig
from question_answer import QuestionAnswer
from document_retriever import DocumentRetriever

# Helper function to load configuration from a YAML file
def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Main function
def main():
    # Load configuration
    config = AppConfig.from_yaml("config.yaml")

    # Ensure the directory exists
    if not os.path.exists(config.vector_store.data_path):
        os.makedirs(config.vector_store.data_path)
        print(f"Created Data irectory: {config.vector_store.data_path}")
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

    # Initialize the document retriever
    document_retriever = DocumentRetriever(config=config)

    # Initialize the QA Wrapper
    question_answer = QuestionAnswer(retriever=document_retriever)

    # ===================================================================================
    # Sample data to add to the vector store (list, pdf, txt)
    # page_content = [
    #     "The bank offers a variety of credit cards with different rewards programs.",
    #     "Our savings accounts provide competitive interest rates.",
    #     "The bank's mortgage plans include fixed and variable interest rate options.",
    #     "Philippines is located in south east asia",
    #     "The capital of the Philippines is Manila",
    #     "chowpin is bald"
    # ]
    # metadata = [
    #     {"source": "bank_product_guide"},
    #     {"source": "bank_website"},
    #     {"source": "mortgage_brochure"},
    #     {"source": "wikipedia"},
    #     {"source": "wikipedia"},
    #     {"source": "barber_shop"}
    # ]

    # Convert to a list of Document objects
    # documents1 = [
    #     Document(page_content=content, metadata=meta)
    #     for content, meta in zip(page_content, metadata)
    # ]

    # Query the retriever and generate responses using RAG
    query_text = "when did James became 2 time nba champion?"
    # # get similarities
    response_similarities = question_answer.generate_similarities_with_score(query_text, top_k=5, filter_score=0.7)

    # get answer from LLM (final format)
    context = utils.format_context(response_similarities)
    response_answer = question_answer.generate_response(query_text, context)

    print(f"\nQuery: {response_answer["query"]}")
    print(f"\nContext: {response_answer["context"]}")
    print(f"\nResult: {response_answer["result"]}")
    print(f"\nSource: {response_answer["source_documents"]}")


if __name__ == "__main__":
    main()

