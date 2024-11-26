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
    # get answer from LLM (final format)
    response_answer, collection = await document_retriever.question_answer(request.query)

    # Extract only `source` and `page` fields
    source = [
        {"source": doc.metadata.get("source", None), "page": doc.metadata.get("page", None)}
        for doc in response_answer["source_documents"]
    ] if response_answer["source_documents"] is not None else None

    return QuestionAnswerResponse(
        query=request.query,
        collection=collection,
        result=response_answer["result"],
        source=source
    )
