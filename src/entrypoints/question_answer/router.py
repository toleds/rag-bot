from application import document_retriever
from domain.model import QuestionAnswerRequest, QuestionAnswerResponse
from fastapi import APIRouter

router = APIRouter(tags=["Question-Answer"])

@router.post("/question-answer")
async def question_answer(request: QuestionAnswerRequest):
    """

    :param request:
    :return:
    """
    # get similarities
    response_similarities = await document_retriever.search(request.query)

    # get answer from LLM (final format)
    response_answer = await document_retriever.question_answer(request.query, response_similarities)

    # Extract only `source` and `page` fields
    source = [
        {"source": doc.metadata.get("source", None), "page": doc.metadata.get("page", None)}
        for doc in response_answer["source_documents"]
    ] if response_answer["source_documents"] is not None else None

    return QuestionAnswerResponse(
        query=request.query,
        result=response_answer["result"],
        source=source
    )
