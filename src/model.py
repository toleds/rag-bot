from pydantic import BaseModel
from langchain_core.documents import Document

class QuestionAnswerRequest(BaseModel):
    query: str

class QuestionAnswerResponse(BaseModel):
    query: str
    result: str
    source: list