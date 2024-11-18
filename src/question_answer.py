class QuestionAnswer:
    """
    RAGWrapper combines a language model with a document retriever
    for Retrieval-Augmented Generation.
    """

    def __init__(self, retriever):
        """
        Init function

        :param retriever:
        """
        self.retriever = retriever

    def generate_response(self, query_text: str, documents:list):
        """
        Generate QA answer

        :param documents:
        :param query_text: The user's input query.
        :return: Generated QA response
        """

        response = self.retriever.question_answer(query_text=query_text, documents=documents)

        return response

    def generate_similarities(self, query_text: str, top_k: int = 2):
        """
        Generate a response using the RAG pipeline.

        :param query_text: The user's input query.
        :param top_k: Number of top documents to retrieve.
        :return: Generated response similarity text.
        """
        response = self.retriever.search(query_text, top_k)

        return response

    def generate_similarities_with_score(self, query_text: str, k: int = 5, filter_score: float = 0.7):
        """
        Generate a response using the RAG pipeline.

        :param k:
        :param filter_score:
        :param query_text: The user's input query.
        :return: Generated response similarity text.
        """
        response = self.retriever.search_with_score(query_text, k, filter_score)

        return response

    def generate_similarities_with_score_no_filter(self, query_text: str, k: int = 5):
        """
        Generate a response using the RAG pipeline.

        :param k:
        :param query_text: The user's input query.
        :return: Generated response similarity text.
        """
        response = self.retriever.search_with_score_no_fiter(query_text, k)

        return response
