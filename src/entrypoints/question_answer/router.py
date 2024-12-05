import asyncio

from starlette.responses import StreamingResponse

from domain.model import QuestionAnswerRequest
from fastapi import APIRouter
from manager import chat_manager

router = APIRouter(tags=["Question-Answer"])


@router.post("/generate")
async def generate(request: QuestionAnswerRequest):
    """

    :param request:
    :return:
    """
    # get answer from LLM (final format)
    response = await chat_manager.process_message(request.query)

    return response


@router.post("/generate-stream")
async def generate_stream(request: QuestionAnswerRequest):
    """

    :param request:
    :return:
    """
    # get answer from LLM (final format)
    response = await chat_manager.process_message(request.query)
    print("\n\n")

    async def response_generator():
        # Simulate chunking by splitting the response into lines
        for line in response.splitlines():
            yield line + "\n"
            await asyncio.sleep(0.1)  # Optional: Simulate delay for streaming effect

    # Return a StreamingResponse
    return StreamingResponse(response_generator(), media_type="text/plain")
