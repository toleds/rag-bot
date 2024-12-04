def format_context(documents, truncate: bool = True):
    """
    reformat document to String

    :param truncate:
    :param documents:
    :return:
    """
    # Use only the top 1 or 2 most relevant documents
    context = "\n\n---\n\n ".join([doc.page_content for doc in documents])

    # Truncate the context if it exceeds max_length
    if len(context) > 500 and truncate:
        context = context[:500]

    return context
