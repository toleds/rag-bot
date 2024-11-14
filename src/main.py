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

    # set openapi key
    os.environ["OPENAI_API_KEY"] = config.llms.openapi_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.llms.hugging_face_key

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

    # documents = utils.extract_text_from_pdf(pdf_path=pdf_path)
    # documents2 = utils.extract_text_from_file(file_path="resources/lebron.txt")
    # ===================================================================================

    # Add documents to the retriever (vector store)
    # print("Adding documents to the vector store...")
    # document_retriever.add_documents(documents1, True)
    # document_retriever.add_documents(documents2, True)

    # Query the retriever and generate responses using RAG
    query_text = "describe chowpin"
    #
    # # get similarities
    response_similarities = question_answer.generate_similarities(query_text)
    print(f"Similarities: {response_similarities}")
    #
    # # get similarities
    # response_similarities_with_score = question_answer.generate_similarities_with_score(query_text)
    # print(f"Similarities with score: {response_similarities_with_score}")

    # get answer from LLM (final format)
    context = utils.format_context(response_similarities)
    print(f"Context: {context}")

    response_answer = question_answer.generate_response(query_text, context)
    print(f"Question: {query_text}")
    print(f"Answer: {response_answer['result'].replace("\n","")}")

    print("End.")


if __name__ == "__main__":
    main()

