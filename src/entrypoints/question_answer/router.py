import asyncio
import re

from starlette.responses import StreamingResponse

from application import llm_service
from domain.model import State
from fastapi import APIRouter

router = APIRouter(tags=["Question-Answer"])


@router.post("/generate")
async def generate(request: State):
    """

    :param request:
    :return:
    """
    # get answer from LLM (final format)
    response_answer, collection = await llm_service.generate(request)
    formatted_response = re.sub(r"\\n", "\n", response_answer["answer"])

    return formatted_response


@router.post("/generate-stream")
async def generate_stream(request: State):
    """

    :param request:
    :return:
    """
    # get answer from LLM (final format)
    response_answer = await llm_service.generate(request)
    formatted_response = re.sub(r"\\n", "\n", response_answer["answer"])

    async def response_generator():
        # Simulate chunking by splitting the response into lines
        for line in formatted_response.splitlines():
            yield line + "\n"
            await asyncio.sleep(0.1)  # Optional: Simulate delay for streaming effect

    # Return a StreamingResponse
    return StreamingResponse(response_generator(), media_type="text/plain")
