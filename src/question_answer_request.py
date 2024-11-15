from pydantic import BaseModel

class QuestionAnswerRequest(BaseModel):
    prompt: str