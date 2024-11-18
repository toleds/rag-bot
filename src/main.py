import os
from contextlib import asynccontextmanager

import utils
import shutil
from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException, status
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
    app.state.document_retriever = document_retriever
    # Initialize the QA Wrapper
    app.state.question_answer = QuestionAnswer(retriever=document_retriever)
    app.state.app_config = config

    print("Application startup")
    yield
    print("Application shutdown")

# Create the FastAPI app
app = FastAPI(title="RAG-Bot", version="1.0.0", base_path="/api/v1", lifespan=lifespan)
# Create a router with a prefix
router = APIRouter(prefix="/api/v1")

@router.post("/question-answer")
async def question_answer(request: QuestionAnswerRequest):
    # get similarities
    response_similarities = app.state.question_answer.generate_similarities_with_score(request.query, filter_score=1.0)

    # get answer from LLM (final format)
    response_answer = app.state.question_answer.generate_response(request.query, response_similarities)

    return QuestionAnswerResponse(
            query=response_answer["query"],
            result=response_answer["result"],
            source=response_answer["source_documents"]
        )


@router.get("/similarity-search")
async  def similarity_search(query: str):
    # get similarities
    response_similarities = app.state.question_answer.generate_similarities_with_score_no_filter(query, top_k=10)

    # Extract document fields and score into a dictionary
    response_data = [
        {
            "document": doc.page_content,        # Assuming the content of the document
            "metadata": doc.metadata,            # Document metadata (like source, author, etc.)
            "score": f"{score}"                  # Similarity score
        }
        for (doc, score) in response_similarities
    ]

    return JSONResponse(content={"similarity_search": response_data}, status_code=status.HTTP_200_OK)


@router.post("/add-document")
async def add_document(file: UploadFile = File(...)):
    file_path = f"{app.state.app_config.vector_store.resource_path}/{file.filename}"
    file_extension = file.filename.split(".")[-1]

    if "txt" in file_extension:
        with open(file_path, "w", encoding="utf8") as buffer:
            buffer.write(file.file.read().decode("utf-8"))

        document = utils.extract_text_from_file(file_path=file_path)
    elif "pdf" in file_extension:
        with open(file_path, "wb",) as buffer:
            shutil.copyfileobj(file.file, buffer)

        document = utils.extract_text_from_pdf(pdf_path=file_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The file extension is not valid.: {file_extension}",
        )

    app.state.document_retriever.add_documents(document, True)

    return JSONResponse(content={"message": "File uploaded successfully.  Processing initiated."}, status_code=status.HTTP_202_ACCEPTED)

@router.post("/initialize-vector-store")
async def initialize_db():
    app.state.document_retriever.initialize_vector_store()
    return {"status": "Vector store initialized!"}

# Include the router into the FastAPI app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
