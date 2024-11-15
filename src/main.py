import os
from contextlib import asynccontextmanager

import utils

from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.responses import JSONResponse

from config import AppConfig
from document_retriever import DocumentRetriever
from question_answer import QuestionAnswer
from model import QuestionAnswerRequest, QuestionAnswerResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load configuration
    config = AppConfig.from_yaml("config.yaml")

    # Ensure the directory exists
    if not os.path.exists(config.vector_store.data_path):
        os.makedirs(config.vector_store.data_path)
        print(f"Created Data directory: {config.vector_store.data_path}")
    else:
        print(f"Using existing Data directory: {config.vector_store.data_path}")

    if not os.path.exists(config.vector_store.resource_path):
        os.makedirs(config.vector_store.resource_path)
        print(f"Created Resource directory: {config.vector_store.resource_path}")
    else:
        print(f"Using existing Resource directory: {config.vector_store.resource_path}")

    # set openapi key
    os.environ["OPENAI_API_KEY"] = config.llms.api_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.llms.api_key

    # Initialize the document retriever
    document_retriever = DocumentRetriever(config=config)

    # Initialize the QA Wrapper
    app.state.question_answer = QuestionAnswer(retriever=document_retriever)

    print("Application startup")
    yield
    print("Application shutdown")

# Create the FastAPI app
app = FastAPI(title="My API", version="1.0.0", base_path="/api/v1", lifespan=lifespan)
# Create a router with a prefix
router = APIRouter(prefix="/api/v1")

@router.post("/question_answer")
async def question_answer(request: QuestionAnswerRequest):
    # get similarities
    response_similarities = app.state.question_answer.generate_similarities_with_score(request.question, top_k=5, filter_score=0.75)

    # get answer from LLM (final format)
    context = utils.format_context(response_similarities)
    response_answer = app.state.question_answer.generate_response(request.question, context)

    return QuestionAnswerResponse(
        question=response_answer["query"],
        answer=response_answer["result"],
        source=response_answer["source_documents"]
    )


@router.post("/add_document")
async def add_document(file: UploadFile = File(...)):
    return JSONResponse({"result": 200})

# Include the router into the FastAPI app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
