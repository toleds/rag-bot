from typing_extensions import Optional, List, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel


class QuestionAnswerRequest(BaseModel):
    query: str


class QuestionAnswerResponse(BaseModel):
    query: str
    collection: str
    result: Optional[str] = None
    source: Optional[list] = None


class State(TypedDict):
    user_id: str
    question: str
    documents: List[Document]
    answer: str
    history: list
