from typing import Optional

from pydantic import BaseModel

class QuestionAnswerRequest(BaseModel):
    query: str

class QuestionAnswerResponse(BaseModel):
    query: str
    collection: str
    result: Optional[str] = None
    source: Optional[list] = None