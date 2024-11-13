class QuestionAnswer:
    """
    RAGWrapper combines a language model with a document retriever
    for Retrieval-Augmented Generation.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def generate_response(self, query_text: str) -> str:
        """
        Generate QA answer

        :param query_text: The user's input query.
        :return: Generated QA response
        """

        response = self.retriever.question_answer(query_text=query_text)

        return response

    def generate_similarities(self, query_text: str, top_k: int = 5):
        """
        Generate a response using the RAG pipeline.

        :param query_text: The user's input query.
        :param top_k: Number of top documents to retrieve.
        :return: Generated response similarity text.
        """
        response = self.retriever.search(query_text, top_k)

        # Print the retrieved documents to inspect their structure
        print(f"Retrieved documents: {response}")

        return response

    def generate_similarities_with_score(self, query_text: str, top_k: int = 5):
        """
        Generate a response using the RAG pipeline.

        :param query_text: The user's input query.
        :param top_k: Number of top documents to retrieve.
        :return: Generated response similarity text.
        """
        response = self.retriever.search_with_score(query_text, top_k)

        # Print the retrieved documents to inspect their structure
        print(f"Retrieved documents: {response}")

        return response
