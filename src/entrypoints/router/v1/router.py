from fastapi import APIRouter

from entrypoints.document_retriever import router as document_retriever
from entrypoints.question_answer import router as question_answer

router = APIRouter(prefix="/v1")
router.include_router(router=question_answer.router)
router.include_router(router=document_retriever.router)
