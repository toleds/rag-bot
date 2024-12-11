import shutil

from fastapi.background import BackgroundTasks

from adapter import document_retriever, llm_service
from common import loaders
from config import config

from fastapi import UploadFile, File, APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Document-Retriever"])


@router.get("/similarity-search")
def similarity_search(query: str):
    """

    :param query:
    :return:
    """
    # get similarities
    documents = document_retriever.retrieve(query)

    # Extract document fields and score into a dictionary
    response_data = [
        {
            "document": doc.page_content,  # Assuming the content of the document
            "metadata": doc.metadata,  # Document metadata (like source, author, etc.)
        }
        for (doc) in documents
    ]

    return JSONResponse(
        content={"similarity_search": response_data},
        status_code=status.HTTP_200_OK,
    )


@router.post("/add-document")
async def add_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """

    :param background_tasks:
    :param file:
    :return:
    """
    file_path = f"{config.vector_store.resource_path}/{file.filename}"
    file_extension = file.filename.split(".")[-1]

    if "txt" in file_extension:
        with open(file_path, "w") as buffer:
            buffer.write(file.file.read().decode("utf-8"))
    elif "pdf" in file_extension:
        with open(
            file_path,
            "wb",
        ) as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The file extension is not valid.: {file_extension}",
        )
    # Schedule background processing
    background_tasks.add_task(_process_document, file_extension, file_path)

    return JSONResponse(
        content={
            "message": "Documents extraction ongoing.  Document embedding ongoing and will be available in a while."
        },
        status_code=status.HTTP_202_ACCEPTED,
    )


@router.post("/add-web-pages")
async def add_web_pages(background_tasks: BackgroundTasks, root_url: str):
    """

    :param root_url:
    :param background_tasks:
    :return:
    """
    # Schedule background processing
    background_tasks.add_task(_process_document, "html", root_url)

    return JSONResponse(
        content={
            "message": "Webpages extraction ongoing.  Document embedding ongoing and will be available in a while."
        },
        status_code=status.HTTP_202_ACCEPTED,
    )


async def _process_document(file_extension: str, path: str):
    """

    :param file_extension:
    :param path:
    :return:
    """
    if "txt" in file_extension:
        documents = await loaders.load_text_file(file_path=path)
    elif "pdf" in file_extension:
        documents = await loaders.load_pdf_with_tables(pdf_path=path)
    elif "html" in file_extension:
        documents = await loaders.load_web_url(path)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The file extension is not valid.: {file_extension}",
        )

    print(f"Document {path} adding to the queue.")
    await document_retriever.add_documents(documents)


@router.post("/switch-collection")
async def create_collection(collection_name: str):
    # init collection
    collection = document_retriever.get_or_create_collection(
        collection_name=collection_name
    )

    # init llm
    llm_service.init_llm()

    return JSONResponse(content={"collection": collection})


@router.get("/list-collection")
async def list_collection():
    collections = document_retriever.get_collection_list()
    collection_dict = [
        {"collection_name": collection.name} for collection in collections
    ]
    return JSONResponse(content=collection_dict)
