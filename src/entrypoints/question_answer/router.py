from starlette.responses import StreamingResponse

from domain.model import QuestionAnswerRequest
from fastapi import APIRouter, Header
from manager import chat_manager

router = APIRouter(tags=["Question-Answer"])


@router.post("/generate")
async def generate(request: QuestionAnswerRequest, x_user_id: str = Header(...)):
    """

    :param x_user_id:
    :param request:
    :return:
    """
    # get answer from LLM (final format)
    response = await chat_manager.process_message(x_user_id, request.query)

    return "\n".join(response.splitlines())


@router.post("/generate-stream")
async def generate_stream(request: QuestionAnswerRequest, x_user_id: str = Header(...)):
    """

    :param x_user_id:
    :param request:
    :return:
    """
    # get answer from LLM (final format)
    response = await chat_manager.process_message(x_user_id, request.query)

    async def response_generator():
        # Simulate chunking by splitting the response into lines
        for line in response.splitlines():
            yield line + "\n"

    # Return a StreamingResponse
    return StreamingResponse(response_generator(), media_type="text/plain")
