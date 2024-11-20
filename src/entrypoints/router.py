from sys import prefix

from fastapi import APIRouter

from entrypoints.document_retriever import router as document_retriever
from entrypoints.question_answer import  router as question_answer

router = APIRouter(prefix="/v1")
router.include_router(prefix="/question-answer", router=question_answer.router)
router.include_router(prefix="/document-retriever", router=document_retriever.router)
