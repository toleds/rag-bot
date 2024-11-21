import asyncio
import shutil

from application import document_retriever
from common import file_utils
from config import config

from fastapi import (
    UploadFile,
    File,
    APIRouter,
    HTTPException,
    status
)
from fastapi.responses import JSONResponse



router = APIRouter(tags=["Document-Retriever"])

@router.get("/similarity-search")
async def similarity_search(query: str):
    """

    :param query:
    :return:
    """
    # get similarities
    response_similarities = await document_retriever.search_with_score_no_fiter(query)

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
    """

    :param file:
    :return:
    """
    file_path = f"{config.vector_store.resource_path}/{file.filename}"
    file_extension = file.filename.split(".")[-1]

    if "txt" in file_extension:
        with open(file_path, "w") as buffer:
            buffer.write(file.file.read().decode("utf-8"))
    elif "pdf" in file_extension:
        with open(file_path, "wb",) as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The file extension is not valid.: {file_extension}",
        )
    # Schedule background processing
    asyncio.create_task(_process_document(file_extension, file_path))

    return JSONResponse(content={"message": "Documents uploaded successfully.  Document embedding ongoing and will be available in a while."},
                        status_code=status.HTTP_202_ACCEPTED)

async def _process_document(file_extension: str, file_path: str):
    document = (
        file_utils.extract_text_from_file(file_path=file_path)
        if "txt" in file_extension
        else file_utils.extract_text_from_pdf(pdf_path=file_path)
    )

    await document_retriever.add_documents(document, True)
    print(f"Document {file_path} added to the queue.")


@router.post("/initialize-vector-store")
async def initialize_db():
    """

    :return:
    """
    document_retriever.initialize_vector_store()
    return {"status": "Vector store initialized!"}
