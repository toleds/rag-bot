class RAGWrapper:
    """
    RAGWrapper combines a language model with a document retriever
    for Retrieval-Augmented Generation.
    """

    def __init__(self, model, retriever):
        self.model = model
        self.retriever = retriever

    def generate_response(self, query_text: str, top_k: int = 5) -> str:
        """
        Generate a response using the RAG pipeline.

        :param query_text: The user's input query.
        :param top_k: Number of top documents to retrieve.
        :return: Generated response text.
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.query(query_text, top_k)

        # Print the retrieved documents to inspect their structure
        print(f"Retrieved documents: {retrieved_docs}")

        """
        [
            {
                'document': [
                    'The bank offers a variety of credit cards with different rewards programs.',
                    'Our savings accounts provide competitive interest rates.',
                    "The bank's mortgage plans include fixed and variable interest rate options."
                ],
                'metadata': [
                    {'source': 'bank_product_guide'},
                    {'source': 'bank_website'},
                    {'source': 'mortgage_brochure'}
                ]
            }
        ]
        """

        try:
            context = "\n\n".join([doc["document"][0] for doc in retrieved_docs])  # Join only the first element if it's a list
        except KeyError as e:
            print(f"KeyError: {e} - Make sure the retrieved docs contain the 'document' key.")
            context = "\n\n".join([str(doc) for doc in retrieved_docs])  # Fallback to string representation


        # Step 3: Generate response using the language model
        prompt = f"Context:\n{context}\n\nQuestion: {query_text}\nAnswer:"
        response = self.model.generate(prompt)

        return response
