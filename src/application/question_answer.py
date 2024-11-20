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
