from pydantic import BaseModel
from langchain_core.documents import Document

class QuestionAnswerRequest(BaseModel):
    question: str

class QuestionAnswerResponse(BaseModel):
    question: str
    answer: str
    source: list[Document]