
def format_context(documents, truncate: bool = False):
    """
    reformat document to String

    :param truncate:
    :param documents:
    :return:
    """
    # Use only the top 1 or 2 most relevant documents
    context = "\n\n---\n\n ".join([doc.page_content for doc in documents])

    # Truncate the context if it exceeds max_length
    if len(context) > 1000 and truncate:
        context = context[:1000]

    return context
