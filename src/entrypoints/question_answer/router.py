import asyncio
import re

from starlette.responses import StreamingResponse

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
    response_answer, collection = await document_retriever.question_answer(
        request.query
    )

    # Extract only `source` and `page` fields
    source = (
        [
            {
                "source": doc.metadata.get("source", None),
                "page": doc.metadata.get("page", None),
            }
            for doc in response_answer["source_documents"]
        ]
        if response_answer["source_documents"] is not None
        else None
    )

    return QuestionAnswerResponse(
        query=request.query,
        collection=collection,
        result=response_answer["result"],
        source=source,
    )


@router.post("/question")
async def question(request: QuestionAnswerRequest):
    """

    :param request:
    :return:
    """
    # get answer from LLM (final format)
    response_answer, collection = await document_retriever.question_answer(
        request.query
    )
    formatted_response = re.sub(r"\\n", "\n", response_answer["result"])

    return formatted_response


@router.post("/question-stream")
async def question_stream(request: QuestionAnswerRequest):
    """

    :param request:
    :return:
    """
    # get answer from LLM (final format)
    response_answer, collection = await document_retriever.question_answer(
        request.query
    )
    formatted_response = re.sub(r"\\n", "\n", response_answer["result"])

    async def response_generator():
        # Simulate chunking by splitting the response into lines
        for line in formatted_response.splitlines():
            yield line + "\n"
            await asyncio.sleep(0.1)  # Optional: Simulate delay for streaming effect

    # Return a StreamingResponse
    return StreamingResponse(response_generator(), media_type="text/plain")
