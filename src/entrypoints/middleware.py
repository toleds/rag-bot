import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.base import RequestResponseEndpoint

def configure(app: FastAPI) -> None:
    app.middleware("http")(_add_process_time_header)
    app.add_middleware(
        CORSMiddleware,
        # The type ignores below are resolved in a newer version of mypy.
        #   We aren't in a position to upgrade our mypy version at this time.
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

async def _add_process_time_header(
            request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response
